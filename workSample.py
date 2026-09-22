"""Approval-gated local work assistant. Dynamic tools use approved templates only."""

import json
import logging
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from llama_cpp import Llama

LOGGER = logging.getLogger(__name__)
Message = dict[str, str]


@dataclass(frozen=True)
class Config:
    base_dir: Path
    work_root: Path
    model_path: Path
    database_path: Path
    proposals_path: Path
    max_context_messages: int = 12
    keep_recent_messages: int = 4

    @classmethod
    def load(cls) -> "Config":
        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env")
        return cls(
            base_dir=base_dir,
            work_root=Path(os.getenv("WORK_ROOT", base_dir)).resolve(),
            model_path=Path(os.getenv("MODEL_PATH", base_dir / "02-Local-AI-and-Vision/faster/Ornith-1.5-9B-Q4_K_M.gguf")),
            database_path=Path(os.getenv("WORK_HISTORY_DB", base_dir / "work_history.db")),
            proposals_path=base_dir / "work_tool_proposals.json",
        )


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class SessionStore:
    """Keeps full searchable history and compact model context per named session."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.session_name = "default"
        self.summary = ""
        self.rolling: list[Message] = []
        self._initialize()
        self.load_session(self.session_name)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.config.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (name TEXT PRIMARY KEY, summary TEXT NOT NULL DEFAULT '', rolling_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_name TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(content, session_name UNINDEXED)")

    def load_session(self, name: str) -> str:
        name = name.strip() or "default"
        with self._connection() as db:
            db.execute("INSERT OR IGNORE INTO sessions (name, updated_at) VALUES (?, ?)", (name, self._now()))
            row = db.execute("SELECT summary, rolling_json FROM sessions WHERE name = ?", (name,)).fetchone()
        self.session_name = name
        self.summary = row["summary"]
        try:
            self.rolling = json.loads(row["rolling_json"])
        except json.JSONDecodeError:
            self.rolling = []
        return f"Loaded session '{name}'."

    def list_sessions(self) -> str:
        with self._connection() as db:
            rows = db.execute("SELECT name, updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
        return json.dumps([dict(row) for row in rows], indent=2)
    def rename_session(self, new_name: str) -> str:
        new_name = new_name.strip()
        if not new_name:
            return "Session name cannot be empty."
        if new_name == self.session_name:
            return f"Session is already named '{new_name}'."
        with self._connection() as db:
            exists = db.execute("SELECT 1 FROM sessions WHERE name = ?", (new_name,)).fetchone()
            if exists:
                return f"A session named '{new_name}' already exists."
            db.execute("UPDATE sessions SET name=?, updated_at=? WHERE name=?", (new_name, self._now(), self.session_name))
            db.execute("UPDATE messages SET session_name=? WHERE session_name=?", (new_name, self.session_name))
            db.execute("UPDATE message_search SET session_name=? WHERE session_name=?", (new_name, self.session_name))
        old_name = self.session_name
        self.session_name = new_name
        return f"Renamed session '{old_name}' to '{new_name}'."

    def record(self, role: str, content: Any, metadata: dict[str, Any] | None = None) -> None:
        text = content if isinstance(content, str) else json.dumps(content, default=str, ensure_ascii=False)
        with self._connection() as db:
            cursor = db.execute(
                "INSERT INTO messages (session_name, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (self.session_name, role, text, json.dumps(metadata or {}, default=str), self._now()),
            )
            db.execute("INSERT INTO message_search (rowid, content, session_name) VALUES (?, ?, ?)", (cursor.lastrowid, text, self.session_name))

    def search(self, query: str, session_name: str = "", limit: int = 5) -> str:
        session = session_name or self.session_name
        try:
            with self._connection() as db:
                rows = db.execute("SELECT m.role, m.content, m.created_at FROM message_search f JOIN messages m ON m.id=f.rowid WHERE f.session_name=? AND f.content MATCH ? ORDER BY m.id DESC LIMIT ?", (session, query, max(1, min(int(limit), 20)))).fetchall()
        except sqlite3.OperationalError:
            with self._connection() as db:
                rows = db.execute("SELECT role, content, created_at FROM messages WHERE session_name=? AND content LIKE ? ORDER BY id DESC LIMIT ?", (session, f"%{query}%", max(1, min(int(limit), 20)))).fetchall()
        return json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2)

    def context(self) -> list[Message]:
        result = []
        if self.summary:
            result.append({"role": "system", "content": f"Session summary:\n{self.summary}"})
        return [*result, *self.rolling]

    def add_chat(self, message: Message) -> None:
        self.rolling.append(message)
        self._save()

    def compact(self, summary: str) -> None:
        self.summary = summary
        self.rolling = self.rolling[-self.config.keep_recent_messages:]
        self._save()

    def _save(self) -> None:
        with self._connection() as db:
            db.execute("UPDATE sessions SET summary=?, rolling_json=?, updated_at=? WHERE name=?", (self.summary, json.dumps(self.rolling, ensure_ascii=False), self._now(), self.session_name))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkTools:
    """Work-safe tools plus approval-gated aliases created from fixed templates."""

    ALLOWED_TEMPLATES = {"read_file", "search_files", "git_diff", "git_status", "run_tests", "sql_query_readonly", "search_history"}

    def __init__(self, config: Config, sessions: SessionStore) -> None:
        self.config = config
        self.sessions = sessions
        self.tools: dict[str, Tool] = {}
        self.proposals = self._load_proposals()
        self._register_static_tools()
        self._activate_approved_proposals()

    def register(self, name: str, description: str, parameters: dict[str, str], handler: Callable[..., str]) -> None:
        self.tools[name] = Tool(name, description, parameters, handler)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(self, call: ToolCall) -> str:
        tool = self.tools.get(call.name)
        if not tool:
            return f"Unknown tool: {call.name}"
        try:
            result = tool.handler(**call.arguments)
            return result if isinstance(result, str) else json.dumps(result, default=str)
        except Exception as error:
            LOGGER.exception("Tool failed: %s", call.name)
            return f"Tool error: {error}"

    def propose_tool(self, name: str, description: str, template: str, parameters: str = "{}") -> str:
        if not name.isidentifier() or name.startswith("_"):
            return "Tool names must be valid public Python identifiers."
        if name in self.tools or any(item["name"] == name for item in self.proposals):
            return f"Tool '{name}' already exists or is pending."
        if template not in self.ALLOWED_TEMPLATES:
            return f"Template must be one of: {', '.join(sorted(self.ALLOWED_TEMPLATES))}."
        try:
            params = json.loads(parameters) if isinstance(parameters, str) else parameters
            if not isinstance(params, dict):
                return "parameters must be a JSON object."
        except json.JSONDecodeError as error:
            return f"Invalid parameters JSON: {error}"
        self.proposals.append({"name": name, "description": description, "template": template, "parameters": params, "status": "pending"})
        self._save_proposals()
        return f"Tool '{name}' proposed. Run /approve-tool {name} to activate it."

    def list_proposals(self) -> str:
        return json.dumps(self.proposals, ensure_ascii=False, indent=2)

    def approve_tool(self, name: str) -> str:
        proposal = next((item for item in self.proposals if item["name"] == name), None)
        if not proposal:
            return f"No proposal named '{name}'."
        proposal["status"] = "approved"
        self._save_proposals()
        self._register_template(proposal)
        return f"Tool '{name}' approved and activated."

    def _register_static_tools(self) -> None:
        self.register("read_file", "Read a text file under WORK_ROOT", {"path": "str"}, self.read_file)
        self.register("search_files", "Search text under WORK_ROOT", {"query": "str"}, self.search_files)
        self.register("git_diff", "Show a Git diff under WORK_ROOT", {}, self.git_diff)
        self.register("git_status", "Show Git status under WORK_ROOT", {}, self.git_status)
        self.register("run_tests", "Run an approved test profile", {"profile": "str"}, self.run_tests)
        self.register("sql_query_readonly", "Run a read-only parameterized SQL Server query", {"query": "str", "parameters": "str"}, self.sql_query_readonly)
        self.register("search_history", "Search the full SQLite session history", {"query": "str", "session_name": "str", "limit": "int"}, self.sessions.search)
        self.register("list_sessions", "List named sessions", {}, self.sessions.list_sessions)
        self.register("load_session", "Load or create a named session", {"name": "str"}, self.sessions.load_session)
        self.register("propose_tool", "Propose a fixed-template tool for human approval", {"name": "str", "description": "str", "template": "str", "parameters": "str"}, self.propose_tool)
        self.register("list_tool_proposals", "List pending and approved tool proposals", {}, self.list_proposals)
        self.register("respond_to_user", "Provide the final response", {"answer": "str"}, lambda answer: answer)

    def _register_template(self, proposal: dict[str, Any]) -> None:
        target = getattr(self, proposal["template"])
        defaults = proposal["parameters"]
        def handler(**arguments: Any) -> str:
            return target(**{**defaults, **arguments})
        self.register(proposal["name"], proposal["description"], {str(k): "str" for k in defaults}, handler)

    def _activate_approved_proposals(self) -> None:
        for proposal in self.proposals:
            if proposal.get("status") == "approved":
                self._register_template(proposal)

    def _load_proposals(self) -> list[dict[str, Any]]:
        if not self.config.proposals_path.is_file():
            return []
        try:
            return json.loads(self.config.proposals_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _save_proposals(self) -> None:
        self.config.proposals_path.write_text(json.dumps(self.proposals, indent=2, ensure_ascii=False), encoding="utf-8")

    def _resolve_path(self, path: str) -> Path:
        resolved = (self.config.work_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if resolved != self.config.work_root and self.config.work_root not in resolved.parents:
            raise ValueError("Path is outside WORK_ROOT.")
        return resolved

    def read_file(self, path: str) -> str:
        return self._resolve_path(path).read_text(encoding="utf-8")

    def search_files(self, query: str) -> str:
        matches = []
        for path in self.config.work_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".cs", ".ts", ".html", ".sql", ".json", ".md"}:
                try:
                    if query.lower() in path.read_text(encoding="utf-8", errors="ignore").lower():
                        matches.append(str(path.relative_to(self.config.work_root)))
                except OSError:
                    pass
                if len(matches) == 30:
                    break
        return json.dumps(matches)

    def git_diff(self) -> str:
        return self._git("diff", "--stat")

    def git_status(self) -> str:
        return self._git("status", "--short")

    def _git(self, *arguments: str) -> str:
        result = subprocess.run(["git", "-C", str(self.config.work_root), *arguments], capture_output=True, text=True, timeout=30)
        return result.stdout or result.stderr or "No output."

    def run_tests(self, profile: str) -> str:
        commands = {
            "python": ["python", "-m", "pytest"],
            "dotnet": ["dotnet", "test"],
            "angular": ["npm", "test", "--", "--watch=false"],
        }
        command = commands.get(profile.lower())
        if not command:
            return f"Unknown test profile. Choose: {', '.join(commands)}."
        result = subprocess.run(command, cwd=self.config.work_root, capture_output=True, text=True, timeout=300)
        return result.stdout if result.returncode == 0 else result.stderr

    @staticmethod
    def sql_query_readonly(query: str, parameters: str = "{}") -> str:
        normalized = query.lower()
        if not (normalized.lstrip().startswith("select") or normalized.lstrip().startswith("with")) or re.search(r"\b(insert|update|delete|merge|drop|alter|create|exec|truncate)\b", normalized):
            return "Blocked: only read-only SELECT/WITH SQL is allowed."
        connection_string = os.getenv("WORK_SQL_CONNECTION_STRING")
        if not connection_string:
            return "Set WORK_SQL_CONNECTION_STRING to enable SQL Server access."
        try:
            import pyodbc
            with pyodbc.connect(connection_string, timeout=15, autocommit=True) as connection:
                cursor = connection.cursor()
                cursor.execute(query, json.loads(parameters))
                columns = [column[0] for column in cursor.description or []]
                return json.dumps([dict(zip(columns, row)) for row in cursor.fetchmany(200)], default=str)
        except Exception as error:
            return f"SQL error: {error}"


class LocalModel:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.instance: Llama | None = None

    def generate(self, messages: list[Message]) -> str:
        if self.instance is None:
            self.instance = Llama(model_path=str(self.config.model_path), n_ctx=8192, verbose=False)
        text = ""
        for chunk in self.instance.create_chat_completion(messages=messages, stream=True, temperature=0.1):
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                print(content, end="", flush=True)
                text += content
        print()
        return text


class Agent:
    def __init__(self, config: Config, sessions: SessionStore, tools: WorkTools) -> None:
        self.config, self.sessions, self.tools = config, sessions, tools
        self.model = LocalModel(config)

    def respond(self, prompt: str) -> None:
        self.sessions.record("user", prompt)
        transient: list[Message] = []  # Tool results stay out of rolling chat context.
        for attempt in range(2):
            messages = [{"role": "system", "content": self._instruction()}, *self.sessions.context(), {"role": "user", "content": prompt}, *transient]
            print("\nAgent > ", end="")
            response = self.model.generate(messages)
            calls = self._parse_calls(response)
            if not calls:
                if attempt == 1:
                    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    self._save_answer(prompt, answer)
                    return
                transient.extend([{"role": "assistant", "content": response}, {"role": "user", "content": "Respond with one JSON tool call only."}])
                continue
            for call in calls:
                result = self.tools.execute(call)
                self.sessions.record("tool", result, {"name": call.name, "arguments": call.arguments})
                if call.name == "respond_to_user":
                    self._save_answer(prompt, result)
                    return
                transient.append({"role": "user", "content": f"Tool result ({call.name}): {result}"})

    def _save_answer(self, prompt: str, answer: str) -> None:
        self.sessions.record("assistant", answer)
        self.sessions.add_chat({"role": "user", "content": prompt})
        self.sessions.add_chat({"role": "assistant", "content": answer})
        if len(self.sessions.rolling) >= self.config.max_context_messages:
            older = self.sessions.rolling[:-self.config.keep_recent_messages]
            summary = "\n".join(f"{item['role']}: {item['content']}" for item in older)[-4000:]
            self.sessions.compact(summary)

    def _instruction(self) -> str:
        return f"You are a work assistant. Use JSON tool calls only. Available tools: {json.dumps(self.tools.schemas())}. Use respond_to_user for final answers. Dynamic tools must be proposed with propose_tool and a human must approve them in the CLI."

    @staticmethod
    def _parse_calls(response: str) -> list[ToolCall]:
        calls = []
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
                calls.append(ToolCall(payload["tool"], payload["args"]))
            elif isinstance(payload, dict) and isinstance(payload.get("answer"), str):
                calls.append(ToolCall("respond_to_user", {"answer": payload["answer"]}))
            index += end
        return calls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = Config.load()
    sessions = SessionStore(config)
    tools = WorkTools(config, sessions)
    agent = Agent(config, sessions, tools)
    print("Commands: /sessions, /session <name>, /rename-session <name>, /tool-proposals, /approve-tool <name>, exit")
    while True:
        try:
            prompt = input(f"[{sessions.session_name}] User > ").strip()
        except EOFError:
            print()
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt == "/sessions":
            print(sessions.list_sessions())
        elif prompt.startswith("/session "):
            print(sessions.load_session(prompt.removeprefix("/session ")))
        elif prompt.startswith("/rename-session "):
            print(sessions.rename_session(prompt.removeprefix("/rename-session ")))
        elif prompt == "/tool-proposals":
            print(tools.list_proposals())
        elif prompt.startswith("/approve-tool "):
            print(tools.approve_tool(prompt.removeprefix("/approve-tool ")))
        elif prompt:
            agent.respond(prompt)


if __name__ == "__main__":
    main()
