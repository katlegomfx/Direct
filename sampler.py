import json
import logging
import os
import re
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler


Message = dict[str, str]
ToolHandler = Callable[..., str]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    model_path: Path
    projector_path: Path
    database_path: Path
    context_size: int = 16_000
    max_agent_turns: int = 10
    max_rolling_messages: int = 12
    keep_recent_messages: int = 4

    @classmethod
    def from_environment(cls) -> "AppConfig":
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env")
        return cls(
            base_dir=base_dir,
            model_path=Path(os.getenv(
                "MODEL_PATH",
                base_dir / "02-Local-AI-and-Vision/faster/Ornith-1.5-9B-Q4_K_M.gguf",
            )),
            projector_path=Path(os.getenv(
                "PROJ_PATH",
                base_dir / "02-Local-AI-and-Vision/faster/mmproj-Ornith-1.5-9B-BF16.gguf",
            )),
            database_path=Path(os.getenv("HISTORY_DB", base_dir / "agent_history.db")),
        )


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Generation:
    content: str
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, str]
    handler: ToolHandler

    def as_prompt_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def as_openai_schema(self) -> dict[str, Any]:
        type_map = {"int": "integer", "float": "number", "bool": "boolean"}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": type_map.get(kind, "string")}
                        for name, kind in self.parameters.items()
                    },
                },
            },
        }


class ConversationStore:
    """SQLite full-history archive plus compact per-session model context."""

    def __init__(self, database_path: Path, max_rolling: int, keep_recent: int) -> None:
        self.database_path = database_path
        self.max_rolling = max_rolling
        self.keep_recent = keep_recent
        self.session_name = "default"
        self.summary = ""
        self.rolling: list[Message] = []
        self._initialize()
        self.load_session(self.session_name)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    name TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    rolling_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS message_search
                    USING fts5(content, session_name UNINDEXED)"""
            )

    def load_session(self, name: str) -> str:
        name = name.strip() or "default"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO sessions
                   (name, summary, rolling_json, created_at, updated_at)
                   VALUES (?, '', '[]', ?, ?)""",
                (name, now, now),
            )
            row = connection.execute(
                "SELECT summary, rolling_json FROM sessions WHERE name = ?", (name,)
            ).fetchone()

        self.session_name = name
        self.summary = row["summary"] or ""
        try:
            rolling = json.loads(row["rolling_json"])
            self.rolling = rolling if isinstance(rolling, list) else []
        except json.JSONDecodeError:
            self.rolling = []
        return f"Loaded session '{name}' with {len(self.rolling)} rolling messages."

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, updated_at, length(summary) AS summary_length FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    def rename_session(self, new_name: str) -> str:
        new_name = new_name.strip()
        if not new_name:
            return "Session name cannot be empty."
        if new_name == self.session_name:
            return f"Session is already named '{new_name}'."
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM sessions WHERE name = ?", (new_name,)).fetchone()
            if exists:
                return f"A session named '{new_name}' already exists."
            connection.execute("UPDATE sessions SET name = ?, updated_at = ? WHERE name = ?", (new_name, self._now(), self.session_name))
            connection.execute("UPDATE messages SET session_name = ? WHERE session_name = ?", (new_name, self.session_name))
            connection.execute("UPDATE message_search SET session_name = ? WHERE session_name = ?", (new_name, self.session_name))
        old_name = self.session_name
        self.session_name = new_name
        return f"Renamed session '{old_name}' to '{new_name}'."

    def model_context(self) -> list[Message]:
        context: list[Message] = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Conversation summary for session '{self.session_name}':\n{self.summary}",
            })
        return [*context, *self.rolling]

    def record_full(self, role: str, content: Any, metadata: dict[str, Any] | None = None) -> None:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (session_name, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (self.session_name, role, text, metadata_json, self._now()),
            )
            connection.execute(
                "INSERT INTO message_search (rowid, content, session_name) VALUES (?, ?, ?)",
                (cursor.lastrowid, text, self.session_name),
            )

    def add_chat_message(self, message: Message) -> None:
        """Store chat memory only; transient tool-loop context is deliberately excluded."""
        self.rolling.append(message)
        self._save_session()

    def needs_compaction(self) -> bool:
        return len(self.rolling) >= self.max_rolling

    def compact(self, summary: str) -> None:
        self.summary = summary.strip()
        self.rolling = self.rolling[-self.keep_recent:]
        self._save_session()

    def search(self, query: str, session_name: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        session_name = session_name or self.session_name
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT m.role, m.content, m.metadata_json, m.created_at
                       FROM message_search f
                       JOIN messages m ON m.id = f.rowid
                       WHERE f.session_name = ? AND f.content MATCH ?
                       ORDER BY m.id DESC LIMIT ?""",
                    (session_name, query, max(1, min(limit, 20))),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError:
            # Fallback keeps ordinary searches useful if the query is not valid FTS syntax.
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT role, content, metadata_json, created_at FROM messages
                       WHERE session_name = ? AND content LIKE ?
                       ORDER BY id DESC LIMIT ?""",
                    (session_name, f"%{query}%", max(1, min(limit, 20))),
                ).fetchall()
            return [dict(row) for row in rows]

    def _save_session(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET summary = ?, rolling_json = ?, updated_at = ? WHERE name = ?",
                (self.summary, json.dumps(self.rolling, ensure_ascii=False), self._now(), self.session_name),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LlmClient:
    """Lazy llama.cpp initialization and normal streamed chat responses."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm: Llama | None = None

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float,
        header: str = "Assistant",
    ) -> Generation:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "tools": tools,
            "tool_choice": "required",
        }
        try:
            return self._consume_stream(kwargs, header)
        except Exception as error:
            # Some local model templates do not support native tool calling.
            LOGGER.warning("Native tool calling failed; using text-tool fallback: %s", error)
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
            return self._consume_stream(kwargs, header)

    def _consume_stream(self, kwargs: dict[str, Any], header: str) -> Generation:
        response = ""
        fragments: dict[int, dict[str, str]] = {}
        print(f"\n{header} > ", end="")
        for chunk in self._get_llm().create_chat_completion(**kwargs):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                print(content, end="", flush=True)
                response += content

            for fragment in delta.get("tool_calls") or []:
                index = fragment.get("index", 0)
                entry = fragments.setdefault(index, {"name": "", "arguments": ""})
                function = fragment.get("function", {})
                entry["name"] += function.get("name", "")
                entry["arguments"] += function.get("arguments", "")
        print()

        calls: list[ToolCall] = []
        for index in sorted(fragments):
            fragment = fragments[index]
            try:
                arguments = json.loads(fragment["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": fragment["arguments"]}
            if fragment["name"] and isinstance(arguments, dict):
                calls.append(ToolCall(fragment["name"], arguments))
        return Generation(response, calls)
    def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Create a concise factual conversation memory. Preserve user goals, decisions, "
                    "file paths, tool outcomes, and unresolved work. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "previous_summary": previous_summary,
                    "messages_to_compress": messages,
                }, ensure_ascii=False, default=str),
            },
        ]
        response = self._get_llm().create_chat_completion(
            messages=prompt,
            stream=False,
            temperature=0.1,
            top_p=0.9,
            max_tokens=512,
        )
        return response["choices"][0]["message"].get("content", "").strip()

    def _get_llm(self) -> Llama:
        if self._llm is None:
            if not self.config.model_path.is_file():
                raise FileNotFoundError(f"Model not found: {self.config.model_path}")
            kwargs: dict[str, Any] = {
                "model_path": str(self.config.model_path),
                "n_ctx": self.config.context_size,
                "verbose": False,
            }
            if self.config.projector_path.is_file():
                kwargs["chat_handler"] = Qwen25VLChatHandler(
                    clip_model_path=str(self.config.projector_path), verbose=False
                )
                kwargs["logits_all"] = True
            self._llm = Llama(**kwargs)
            logging.getLogger("llama-cpp-python").setLevel(logging.CRITICAL + 1)
        return self._llm


class ToolRegistry:
    """One registration supplies both implementation and model-visible metadata."""

    def __init__(self, memory: ConversationStore) -> None:
        self.memory = memory
        self._tools: dict[str, ToolDefinition] = {}
        self._register_defaults()

    @property
    def prompt_schema(self) -> list[dict[str, Any]]:
        return [tool.as_prompt_schema() for tool in self._tools.values()]

    @property
    def openai_schema(self) -> list[dict[str, Any]]:
        return [tool.as_openai_schema() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> str:
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Unknown tool: {call.name}"
        print(f"--- [System: Executing {call.name}] ---")
        try:
            result = tool.handler(**call.arguments)
        except Exception as error:
            LOGGER.exception("Tool failed: %s", call.name)
            result = f"Error executing {call.name}: {error}"
        print(result)
        return result

    def _register(self, name: str, description: str, parameters: dict[str, str], handler: ToolHandler) -> None:
        self._tools[name] = ToolDefinition(name, description, parameters, handler)

    def _register_defaults(self) -> None:
        self._register("read_file", "Read a UTF-8 text file", {"path": "str"}, self.read_file)
        self._register("write_file", "Write a UTF-8 text file", {"path": "str", "content": "str"}, self.write_file)
        self._register("execute_python", "Execute Python code", {"code": "str"}, self.execute_python)
        self._register("run_command", "Run a shell command", {"command": "str"}, self.run_command)
        self._register("search_history", "Search full SQLite conversation history", {"query": "str", "session_name": "str", "limit": "int"}, self.search_history)
        self._register("list_sessions", "List named conversation sessions", {}, self.list_sessions)
        self._register("load_session", "Switch to or create a named conversation session", {"name": "str"}, self.load_session)
        self._register("respond_to_user", "Provide the final response to the user", {"answer": "str"}, self.respond_to_user)

    @staticmethod
    def read_file(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as error:
            return f"Error reading file: {error}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            Path(path).write_text(content, encoding="utf-8")
            return "Success"
        except OSError as error:
            return f"Error writing file: {error}"

    @staticmethod
    def execute_python(code: str) -> str:
        try:
            result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=60)
            return result.stdout if result.returncode == 0 else result.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Error during Python execution: {error}"

    @staticmethod
    def run_command(command: str) -> str:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            return result.stdout if result.returncode == 0 else result.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            return f"Error during command execution: {error}"

    def search_history(self, query: str, session_name: str = "", limit: int = 5) -> str:
        matches = self.memory.search(query, session_name or None, int(limit))
        if not matches:
            return "No matching history found."
        return json.dumps(matches, ensure_ascii=False, indent=2)

    def list_sessions(self) -> str:
        return json.dumps(self.memory.list_sessions(), ensure_ascii=False, indent=2)

    def load_session(self, name: str) -> str:
        return self.memory.load_session(name)

    @staticmethod
    def respond_to_user(answer: str) -> str:
        return answer


class Agent:
    FINAL_TOOL = "respond_to_user"
    XML_FUNCTION = re.compile(r"<function=(?P<name>[\w_]+)>(?P<body>.*?)</function>", re.DOTALL)
    XML_PARAMETER = re.compile(r"<parameter=(?P<name>[\w_]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL)

    def __init__(self, config: AppConfig, llm: LlmClient, memory: ConversationStore, tools: ToolRegistry) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.tools = tools

    def respond(self, prompt: str) -> None:
        self.memory.record_full("user", prompt)
        tool_context: list[Message] = []  # Per-turn only: never added to rolling chat memory.
        session_at_start = self.memory.session_name

        for _ in range(self.config.max_agent_turns):
            messages = self._build_messages(prompt, tool_context)
            generation = self.llm.stream_chat(
                messages, self.tools.openai_schema, temperature=0.1, header="Thinker"
            )
            response = generation.content
            calls = generation.tool_calls or self._parse_tool_calls(response)

            if not calls:
                if tool_context and response.strip():
                    # Keep the CLI responsive when a model cannot follow tool syntax.
                    answer = self._strip_thinking(response)
                    self.memory.record_full("assistant", answer, {"fallback": "plain_text"})
                    self._save_chat_turn(prompt, answer)
                    return
                tool_context.extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": self._repair_instruction()},
                ])
                continue

            for call in calls:
                self.memory.record_full("assistant", response, {"tool_call": call.name, "arguments": call.arguments})
                result = self.tools.execute(call)
                self.memory.record_full("tool", result, {"tool_name": call.name})

                if call.name == self.FINAL_TOOL:
                    self._save_chat_turn(prompt, result)
                    return

                if call.name == "load_session" and self.memory.session_name != session_at_start:
                    session_at_start = self.memory.session_name
                    tool_context = [{"role": "user", "content": f"Session switched. Tool result: {result}"}]
                    break

                tool_context.extend([
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": f"Tool result for {call.name}: {result}"},
                ])

        LOGGER.warning("Agent stopped after reaching the tool-call limit.")

    def _save_chat_turn(self, prompt: str, answer: str) -> None:
        self.memory.add_chat_message({"role": "user", "content": prompt})
        self.memory.add_chat_message({"role": "assistant", "content": answer})
        self.memory.record_full("assistant", answer, {"final": True})

        if self.memory.needs_compaction():
            older_messages = self.memory.rolling[:-self.config.keep_recent_messages]
            try:
                summary = self.llm.summarize(self.memory.summary, older_messages)
            except Exception as error:
                LOGGER.warning("Could not summarize history: %s", error)
                summary = self.memory.summary
            if summary:
                self.memory.compact(summary)

    def _build_messages(self, prompt: str, tool_context: list[Message]) -> list[Message]:
        instruction = (
            f"Available tools: {json.dumps(self.tools.prompt_schema)}. "
            "Use one or more tool calls only; do not write a plain-text answer. "
            f"Use {self.FINAL_TOOL} to answer the user. "
            'JSON format: {"tool": "tool_name", "args": {}}. '
            "You can use multiple JSON tool objects in one response."
        )
        return [
            {"role": "system", "content": f"You are an AI agent. {instruction}"},
            *self.memory.model_context(),
            {"role": "user", "content": prompt},
            *tool_context,
        ]

    def _repair_instruction(self) -> str:
        return (
            "Respond only with a tool call. For a final response use: "
            '{"tool": "respond_to_user", "args": {"answer": "your answer"}}.'
        )

    @staticmethod
    def _strip_thinking(response: str) -> str:
        return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    @classmethod
    def _parse_tool_calls(cls, response: str) -> list[ToolCall]:
        found: list[tuple[int, ToolCall]] = []
        decoder = json.JSONDecoder()
        index = 0
        while index < len(response):
            if response[index] != "{":
                index += 1
                continue
            try:
                payload, end = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(payload, dict) and isinstance(payload.get("tool"), str) and isinstance(payload.get("args", {}), dict):
                found.append((index, ToolCall(payload["tool"], payload.get("args", {}))))
            elif isinstance(payload, dict) and isinstance(payload.get("answer"), str):
                # Tolerate models that omit the wrapper but provide a valid final payload.
                found.append((index, ToolCall(cls.FINAL_TOOL, {"answer": payload["answer"]})))
            index += end

        for match in cls.XML_FUNCTION.finditer(response):
            arguments = {
                item.group("name"): item.group("value").strip()
                for item in cls.XML_PARAMETER.finditer(match.group("body"))
            }
            found.append((match.start(), ToolCall(match.group("name"), arguments)))
        return [call for _, call in sorted(found, key=lambda item: item[0])]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = AppConfig.from_environment()
    memory = ConversationStore(config.database_path, config.max_rolling_messages, config.keep_recent_messages)
    agent = Agent(config, LlmClient(config), memory, ToolRegistry(memory))

    print("Commands: /sessions, /session <name>, /rename-session <name>, exit")
    while True:
        try:
            prompt = input(f"[{memory.session_name}] User > ").strip()
        except EOFError:
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        if prompt == "/sessions":
            print(json.dumps(memory.list_sessions(), ensure_ascii=False, indent=2))
            continue
        if prompt.startswith("/session "):
            print(memory.load_session(prompt.removeprefix("/session ")))
            continue
        if prompt.startswith("/rename-session "):
            print(memory.rename_session(prompt.removeprefix("/rename-session ")))
            continue
        agent.respond(prompt)


if __name__ == "__main__":
    main()
