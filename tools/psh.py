"""PSH Simple - a line-by-line shell for agentic objects with prompt_toolkit."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from typing import Any, Callable

from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from peteos import AgenticObject, Error, tool

RESET = "\033[0m"
BOLD = "\033[1m"

TAG_COLORS: dict[str, str] = {
    "INVOKE": "\033[36m",
    "TOOL": "\033[33m",
    "RUN": "\033[35m",
    "RET": "\033[32m",
    "ERR": "\033[31m",
    "USER": "\033[34m",
    "AGENT": "\033[92m",
    "ERROR": "\033[91m",
    "RESULT": "\033[96m",
    "SHELL": "\033[93m",
    "TOOLS": "\033[90m",
    "CTX": "\033[97m",
    "SYSTEM": "\033[97m",
    "FUNC": "\033[96m",
    "SESSION": "\033[96m",
    "HINT": "\033[90m",
}


def _make_ctrl_c_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add(Keys.ControlC)
    def _ctrl_c(event: Any) -> None:
        if not event.current_buffer.text:
            raise KeyboardInterrupt
        event.current_buffer.text = ""

    return kb


@dataclass
class _PromptMsg:
    text: str


@dataclass
class _FuncCallMsg:
    fname: str
    fargs: tuple
    fkwargs: dict


def _c(tag: str, text: str, hook: str = "") -> str:
    color = TAG_COLORS.get(tag, "")
    hook_info = f"<{hook}>" if hook else ""
    return f"{color}{BOLD}[{tag}]{hook_info} {text}{RESET}" if color else f"[{tag}]{hook_info} {text}"

_FUNC_COLOR = "\033[93m"
_ARG_COLOR = "\033[96m"
_TYPE_COLOR = "\033[35m"

def _fmt_tool(t: Any) -> str:
    parts: list[str] = []
    for pname, pinfo in (t.parameters or {}).items():
        ptype = pinfo.get("type", "any")
        parts.append(f"{_ARG_COLOR}{BOLD}{pname}{RESET}: {_TYPE_COLOR}{ptype}{RESET}")
    params = ", ".join(parts)
    return f"{_FUNC_COLOR}{BOLD}{t.name}{RESET}({params})"


def _fmt_json(s: str) -> str:
    """Try to parse a JSON string and pretty-print it, else return as-is."""
    try:
        parsed = json.loads(s)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return s


def _fmt_tool_args(args_str: str) -> str:
    """Format tool call arguments with nested JSON in string values expanded.

    Outer layer: JSON object (dict) parsed and formatted as key = value pairs.
    Nested layer: string values that are themselves valid JSON are parsed and
    pretty-printed as sub-blocks.
    String values that are not JSON are shown raw.
    """
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        try:
            import ast
            evaluated = ast.literal_eval(args_str)
            reserialized = json.dumps(evaluated, ensure_ascii=False, indent=2)
            return _fmt_tool_args(reserialized)
        except Exception:
            return args_str

    if not isinstance(args, dict):
        return json.dumps(args, ensure_ascii=False, indent=2)

    if not args:
        return "()"

    lines: list[str] = []
    for key, val in args.items():
        if isinstance(val, str):
            try:
                nested = json.loads(val)
                nested_str = json.dumps(nested, ensure_ascii=False, indent=4)
                indented = "\n".join("      " + ln for ln in nested_str.splitlines())
                lines.append(f"  {key} =\n{indented}")
            except (json.JSONDecodeError, TypeError):
                if "\n" in val:
                    indented = "\n".join("      " + ln for ln in val.splitlines())
                    lines.append(f"  {key} =\n{indented}")
                else:
                    lines.append(f"  {key} = {val}")
        else:
            formatted = json.dumps(val, ensure_ascii=False, indent=2)
            indented = "\n".join("      " + ln for ln in formatted.splitlines())
            lines.append(f"  {key} =\n{indented}")

    return "(\n" + "\n".join(lines) + "\n)"


class PSH:
    """A simple line-by-line shell wrapping a Peteos AgenticObject.

    Usage: enter a prompt, get answers, hooks print as they fire.
    Slash commands (prefix /):
        /quit          exit the shell
        /tools         list @tool methods on the wrapped agent
        /session       show current session ID
        /session <id>  switch to session <id> (persistent conversation)
        /help          show help

    Hooks (injected on every invoke_agent call):
        on_invoke              → [INVOKE] prompt
        on_tool_call           → [TOOL] tool_name
        before_tool_execution  → [RUN] tool(args)
        after_tool_execution   → [RET] tool → result  |  [ERR] tool: reason
        after_message_append   → [USER] / [AGENT] message content
        on_invoke_complete     → [RESULT] value  |  [ERROR] message
    """

    _TITLE = "=== PSH Context ==="

    _PYTHON_COMPLETER = WordCompleter([
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "not", "or", "pass",
        "raise", "return", "try", "while", "with", "yield",
        "True", "False", "None",
        "print", "len", "range", "list", "dict", "str", "int", "float", "bool",
        "tuple", "set", "type", "isinstance", "open", "enumerate", "zip",
        "map", "filter", "sorted", "reversed", "abs", "min", "max", "sum",
        "any", "all", "round", "getattr", "hasattr", "setattr", "delattr",
        "exec", "compile", "eval", "dir", "vars", "help", "id", "repr",
        "agents", "spawn", "terminate", "call",
    ], ignore_case=True)

    def __init__(self, agents: dict[str, AgenticObject], message_queue: queue.Queue, result_queue: queue.Queue[Any], state: _ShellState) -> None:
        super().__init__()
        self._agents = agents
        self._message_queue = message_queue
        self._result_queue = result_queue
        self._state = state
        self._session = PromptSession(
            history=FileHistory(os.path.join(os.getcwd(), ".psh_history")),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=_make_ctrl_c_bindings(),
        )
        self._agent_session = PromptSession(
            history=FileHistory(os.path.join(os.getcwd(), ".psh_agent_history")),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=_make_ctrl_c_bindings(),
        )

    @property
    def _agent(self) -> AgenticObject:
        return self._agents[self._state.foreground_agent_name]

    def _prompt_str(self) -> str:
        return "?> " if self._state.mode == "agent" else "!> "

    def _session_for(self, mode: str) -> PromptSession:
        return self._session if mode == "code" else self._agent_session

    def _get_prompt_args(self, mode: str) -> dict[str, Any]:
        base = "?> " if mode == "agent" else "!> "
        if self._state._allow_all_session:
            message: Any = HTML(f"<ansired><b>{base}</b></ansired>")
        elif self._state._deny_all_session:
            message = HTML(f"<ansicyan><b>{base}</b></ansicyan>")
        elif not self._state.ask_confirmation:
            message = HTML(f"<ansired><b>{base}</b></ansired>")
        else:
            message = HTML(f"<b>{base}</b>")
        result: dict[str, Any] = {"message": message}
        if mode == "code":
            result["multiline"] = True
            result["completer"] = self._PYTHON_COMPLETER
            result["complete_while_typing"] = True
            result["prompt_continuation"] = lambda width, ln, soft: ".  "
        return result

    async def run_shell(self) -> str:
        print(f"{self._TITLE}")
        print(_c("HINT", "/help commands, /quit exit  |  ? agent mode  ! code mode  Ctrl+C clear line"))
        while True:
            session = self._session_for(self._state.mode)
            args = self._get_prompt_args(self._state.mode)
            try:
                with patch_stdout():
                    raw = await session.prompt_async(**args)
            except (EOFError, KeyboardInterrupt):
                print("\n[SHELL] EOF — bye")
                break
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("/"):
                if self._shell_cmd(raw):
                    break
            elif raw == "?":
                self._state.mode = "agent"
            elif raw == "!":
                self._state.mode = "code"
                print(_c("HINT", "Code mode: Enter=newline, Meta+Enter=execute, Alt+F=accept word, →/Ctrl+E=accept full"))
            elif raw.startswith("?"):
                self._invoke_once(raw)
            elif raw.startswith("!"):
                self._code_once(raw)
            else:
                if self._state.mode == "agent":
                    self._message_queue.put(_PromptMsg(raw))
                    self._result_queue.get()
                else:
                    self._code_once_raw(raw)
        return "Shell closed."

    def _invoke_once(self, raw: str) -> None:
        if len(raw) < 2:
            return
        text = raw[1:]
        while text.startswith("??"):
            text = "?" + text[2:]
        if not text:
            return
        self._message_queue.put(_PromptMsg(text))
        self._result_queue.get()

    def _code_once(self, raw: str) -> None:
        if len(raw) < 2:
            return
        self._run_code(raw[1:])

    def _code_once_raw(self, raw: str) -> None:
        self._run_code(raw)

    def _run_code(self, code: str) -> None:
        try:
            compiled = compile(code, "<shell>", "exec")
        except SyntaxError as e:
            print(_c("ERROR", f"SyntaxError: {e}"))
            return
        try:
            exec(compiled, self._state.code_globals)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            for line in tb.splitlines():
                print(_c("ERROR", line))
            return
        for key, val in self._state.code_globals.items():
            if key.startswith("_"):
                continue
            if key in self._state._func_seen:
                continue
            if not callable(val):
                continue
            if key in ("spawn", "terminate", "list_classes", "agents", "call"):
                continue
            self._state._func_registry[key] = val
            self._state._func_seen.add(key)
            doc = getattr(val, "__doc__", None) or ""
            print(_c("SHELL", f"Registered /{key} — {doc.strip().split(chr(10))[0]}"))

    def _shell_cmd(self, raw: str) -> bool:
        parts = raw.lstrip("/").split()
        cmd = parts[0].lower() if parts else ""

        if cmd == "quit":
            print(_c("SHELL", "/quit"))
            return True
        elif cmd == "tools":
            self._list_tools()
        elif cmd == "session":
            self._session_cmd(parts[1:] if len(parts) > 1 else None)
        elif cmd == "ask_confirmation":
            if self._state._allow_all_session or self._state._deny_all_session:
                self._state.ask_confirmation = True
                self._state._allow_all_session = False
                self._state._deny_all_session = False
                print(_c("SHELL", "Ask for tool confirmation: ON"))
            else:
                self._state.ask_confirmation = not self._state.ask_confirmation
                state = "ON" if self._state.ask_confirmation else "OFF"
                print(_c("SHELL", f"Ask for tool confirmation: {state}"))
        elif cmd == "set_output":
            if len(parts) < 3:
                flags = ", ".join(f"{k}={v}" for k, v in self._state.output_flags.items())
                print(_c("SHELL", f"Output flags: {flags}"))
            else:
                key = parts[1].upper()
                val = parts[2].lower() in ("1", "true", "yes", "on")
                if key in self._state.output_flags:
                    self._state.output_flags[key] = val
                    print(_c("SHELL", f"Output {key}={val}"))
                else:
                    print(_c("ERROR", f"Unknown output key: {key}"))
        elif cmd == "agent":
            self._agent_cmd(parts[1:] if len(parts) > 1 else None)
        elif cmd == "help":
            print(
                _c("SHELL",
                "/help\n"
                "  PSH — Peteos Shell\n"
                "  /quit   exit shell\n"
                "  /tools  list agent tools\n"
                "  /session  show current session ID\n"
                "  /session <id>  switch to session <id>\n"
                "  /session none  anonymous session (no memory)\n"
                "  /agent  show current agent name\n"
                "  /agent <name>  switch to agent <name>\n"
                "  /ask_confirmation  toggle tool confirmation\n"
                "  /set_output  show all output flags\n"
                "  /set_output <key> <on>  set output flag (on: 1/true/yes/on, off: 0/false/no/off)\n"
                "  /funcs  list registered functions\n"
                "  /help   this message\n"
                "\n"
                "  MODES\n"
                "  Agent mode  (prompt: ?> )  — talk to the foreground agent.\n"
                "  Code mode   (prompt: !> )  — execute Python in a sandbox with agents.\n"
                "  ?  alone → switch to agent mode.  !  alone → switch to code mode.\n"
                "  ?<text>  in any mode: briefly invoke agent, stay in current mode.\n"
                "  !<code>  in any mode: briefly execute code, stay in current mode.\n"
                "\n"
                "  AGENT MODE  (prompt: ?> )\n"
                "  ?<text>  invoke agent (escape leading ?? for literal ? at start)\n"
                "  <text>  invoke agent (shorthand, same as ?)\n"
                "\n"
                "  CODE MODE  (prompt: !> )\n"
                "  !<code>  execute Python code\n"
                "  <code>  execute Python code (shorthand, same as !)\n"
                "  Available in code mode:\n"
                "    spawn(name, cls_or_obj)  spawn/register an agent\n"
                "    terminate(name)         remove agent from registry\n"
                "    list_classes()          list known agentic classes\n"
                "    agents                 dict of live agent instances\n"
                "    print, len, range, list, dict, str, int, float, bool\n"
                "    type, isinstance, open, map, filter, sorted, zip\n"
                "    + all standard literals and operators\n"
                "\n"
                "  Functions defined in code mode (no leading _) are registered\n"
                "  as slash commands and run in the worker thread.\n"
                ))
        elif cmd == "funcs":
            if not self._state._func_registry:
                print(_c("SHELL", "No registered functions yet (define in code mode)"))
            else:
                print(_c("SHELL", f"{len(self._state._func_registry)} registered function(s):"))
                for fname in self._state._func_registry:
                    f = self._state._func_registry[fname]
                    doc = getattr(f, "__doc__", None) or ""
                    print(f"  /{fname}  — {doc.strip().split(chr(10))[0]}")
        elif cmd in self._state._func_registry:
            fname = cmd
            raw_args = parts[1:] if len(parts) > 1 else []
            self._message_queue.put(_FuncCallMsg(fname, tuple(raw_args), {}))
            try:
                rfname, stdout_val, err_msg, result = self._state.func_result_queue.get(timeout=30)
            except queue.Empty:
                print(_c("ERROR", f"Timed out waiting for /{fname}"))
            else:
                display = f"{fname}{raw_args if raw_args else '()'}"
                print(_c("RUN", display), end="")
                if stdout_val:
                    print(stdout_val, end="")
                if err_msg:
                    print(_c("ERR", err_msg))
                else:
                    print(_c("RET", result if isinstance(result, str) else repr(result)))
        else:
            print(_c("SHELL", f"Unknown command: /{cmd}"))
        return False

    def _session_cmd(self, args: list[str] | None) -> None:
        if args:
            if args[0].lower() == "none":
                self._state.thread_id = None
                self._state.first_invoke = True
                print(_c("SESSION", "Session: none — anonymous (no memory)"))
            else:
                self._state.thread_id = args[0]
                self._state.first_invoke = True
                print(_c("SESSION", f"Switched to session: {self._state.thread_id}"))
        else:
            if self._state.thread_id:
                print(_c("SESSION", f"Current session: {self._state.thread_id}"))
            else:
                print(_c("SESSION", "Session: none — anonymous (no memory)"))

    def _agent_cmd(self, args: list[str] | None) -> None:
        if args:
            name = args[0]
            if name in self._agents:
                self._state.foreground_agent_name = name
                self._state.first_invoke = True
                print(_c("AGENT", f"Switched to agent: {name}"))
            else:
                print(_c("ERROR", f"Unknown agent: {name}"))
        else:
            print(_c("AGENT", f"Current agent: {self._state.foreground_agent_name}"))

    def _list_tools(self) -> None:
        tools = self._agent._oap_tool_manager.get_tool_list()
        print(_c("TOOLS", f"{len(tools)} tool(s):"))
        for t in tools:
            print(f"  {_fmt_tool(t)}  —  {t.description}")


class _DemoAgent(AgenticObject):
    """A demo agent with two simple tools."""

    @tool
    def hello(self, name: str) -> str:
        """Greet the named person."""
        return f"Hello, {name}!"

    @tool
    def add(self, a: float, b: float) -> float:
        """Add two numbers and return the sum."""
        return a + b


class _ShellState:
    def __init__(self) -> None:
        self.first_invoke = True
        self.thread_id: str | None = "default"
        self.ask_confirmation = True
        self.foreground_agent_name: str = "default"
        self.mode: str = "agent"
        self.code_globals: dict[str, Any] = {}
        self.prompt_queue: queue.Queue[str | None] = queue.Queue()
        self.result_queue: queue.Queue[Any] = queue.Queue()
        self.func_result_queue: queue.Queue[Any] = queue.Queue()
        self._func_registry: dict[str, Callable] = {}
        self._func_seen: set[str] = set()
        self.output_flags: dict[str, bool] = {
            "RUN": True,
            "TOOL": True,
            "DENY": True,
            "SESSION": True,
            "AGENT": True,
            "SHELL": True,
            "ERROR": True,
            "TOOLS": True,
            "RESULT": True,
            "READ": True,
        }
        self._allow_all_remaining = False
        self._deny_all_remaining = False
        self._allow_all_session = False
        self._deny_all_session = False


def _spawn(name: str, agent_or_cls: Any, agents: dict[str, Any]) -> None:
    if isinstance(agent_or_cls, str):
        import importlib
        module_path, class_name = agent_or_cls.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        agent_or_cls = getattr(mod, class_name)()
    if hasattr(agent_or_cls, "invoke_agent"):
        agents[name] = agent_or_cls
        print(_c("AGENT", f"Spawned agent {name!r}"))
    else:
        raise TypeError(f"{name!r} is not an AgenticObject (has no invoke_agent)")


def _terminate(name: str, agents: dict[str, Any]) -> None:
    if name in agents:
        del agents[name]
        print(_c("AGENT", f"Terminated agent {name!r}"))


def _list_classes() -> None:
    pass


def _make_code_globals(agents: dict[str, Any], state) -> dict[str, Any]:
    import builtins as _b
    return {
        "__builtins__": {
            **vars(_b),
            "__import__": __import__,
        },
        "agents": agents,
        "spawn": lambda name, obj: _spawn(name, obj, agents),
        "terminate": lambda name: _terminate(name, agents),
        "list_classes": _list_classes,
        "ask_confirmation": lambda on: setattr(state, "ask_confirmation", bool(on)),
        "agent": lambda name: setattr(state, "foreground_agent_name", name),
        "session": lambda name: setattr(state, "thread_id", name if name else None),
        "set_output": lambda key, on: state.output_flags.update({key: bool(on)}),
    }


def _make_bte(state: _ShellState) -> Callable[[Any], Any]:
    async def _bte(tc: Any) -> None:
        n = getattr(tc, "name", "?") if tc else "?"
        raw = getattr(tc, "raw_dict", None)
        args_str = raw.get("arguments", "{}") if raw else "{}"
        if state.output_flags.get("RUN", True):
            print(_c("RUN", f"{n}{_fmt_tool_args(args_str)}"))
        if not state.ask_confirmation:
            return
        if state._allow_all_session:
            return
        if state._deny_all_session:
            if state.output_flags.get("DENY", True):
                print(_c("DENY", n))
            return (False, f"Tool '{n}' denied by user.")
        if state._allow_all_remaining:
            return
        if state._deny_all_remaining:
            if state.output_flags.get("DENY", True):
                print(_c("DENY", n))
            return (False, f"Tool '{n}' denied by user.")
        loop = asyncio.get_running_loop()
        raw_answer = await loop.run_in_executor(
            None, lambda:             input("  allow? [y/n/yes/no/msg] | [Y/N] invoke | [YES/NO] session (or type custom msg): ").strip()
        )
        if raw_answer == "Y":
            state._allow_all_remaining = True
            return
        if raw_answer == "N":
            if state.output_flags.get("DENY", True):
                print(_c("DENY", n))
            state._deny_all_remaining = True
            return (False, f"Tool '{n}' denied by user.")
        if raw_answer == "YES":
            state._allow_all_session = True
            state.ask_confirmation = False
            return
        if raw_answer == "NO":
            if state.output_flags.get("DENY", True):
                print(_c("DENY", n))
            state._deny_all_session = True
            return (False, f"Tool '{n}' denied by user.")
        answer_lower = raw_answer.lower()
        if answer_lower in ("y", "yes", ""):
            return
        if answer_lower in ("n", "no"):
            if state.output_flags.get("DENY", True):
                print(_c("DENY", n))
            return (False, f"Tool '{n}' denied by user.")
        if state.output_flags.get("DENY", True):
            print(_c("DENY", n))
        return (False, raw_answer)
    return _bte


def _make_ate(state: _ShellState) -> Callable[[Any, Any, Any, bool], None]:
    def _hook(runner: Any, tc: Any, res: Any, ok: bool) -> None:
        n = getattr(tc, "name", "?") if tc else "?"
        res_str = str(res) if res else ""
        prefix = "RET" if ok else "ERR"
        if state.output_flags.get(prefix, True):
            print(_c(prefix, f"{n} -> {_fmt_tool_args(res_str)}"))
    return _hook


def _make_mappend(state: _ShellState) -> Callable[[Any, Any], None]:
    def _hook(runner: Any, msg: Any) -> None:
        printable = getattr(msg, "printable", None)
        role = getattr(msg, "role", "?").upper()
        if printable:
            text = printable()
            if text and state.output_flags.get("READ", True):
                print(_c(role, text))
                return
        raw = getattr(msg, "content", "")
        if isinstance(raw, list):
            if state.output_flags.get("READ", True):
                print(_c(role, repr(raw)))
        elif isinstance(raw, str) and raw:
            if state.output_flags.get("READ", True):
                print(_c(role, raw))
    return _hook


def _make_done(state: _ShellState) -> Callable[[dict], None]:
    def _hook(ctx: dict) -> None:
        state._allow_all_remaining = False
        state._deny_all_remaining = False
        r = ctx.get("result")
        if isinstance(r, Error):
            if state.output_flags.get("ERROR", True):
                print(_c("ERROR", r.message))
        elif r is not None:
            if state.output_flags.get("RESULT", True):
                print(_c("RESULT", r if isinstance(r, str) else repr(r)))
        else:
            if state.output_flags.get("RESULT", True):
                print(_c("RESULT", "None"))
    return _hook


def _mappend(runner: Any, msg: Any) -> None:
    printable = getattr(msg, "printable", None)
    role = getattr(msg, "role", "?").upper()
    if printable:
        text = printable()
        if text:
            print(_c(role, text, hook="_mappend"))
            return
    raw = getattr(msg, "content", "")
    if isinstance(raw, list):
        print(_c(role, repr(raw), hook="_mappend"))
    elif isinstance(raw, str) and raw:
        print(_c(role, raw, hook="_mappend"))


def _done(ctx: dict) -> None:
    r = ctx.get("result")
    if isinstance(r, Error):
        print(_c("ERROR", r.message))
    elif r is not None:
        print(_c("RESULT", r if isinstance(r, str) else repr(r)))
    else:
        print(_c("RESULT", "None"))


def _before_llm(runner: Any, ctx: Any, state: _ShellState) -> None:
    if not state.first_invoke:
        return
    state.first_invoke = False
    for msg in ctx.messages:
        role = getattr(msg, "role", "?").upper()
        printable = getattr(msg, "printable", None)
        if printable:
            text = printable()
            if text:
                print(_c(role, text, hook="_before_llm"))
                continue
        raw = getattr(msg, "content", "")
        if isinstance(raw, list):
            print(_c(role, repr(raw), hook="_before_llm"))
        elif isinstance(raw, str) and raw:
            print(_c(role, raw, hook="_before_llm"))


def _peteos_worker(
    agents: dict[str, AgenticObject],
    message_queue: queue.Queue,
    result_queue: queue.Queue[Any],
    state: _ShellState,
) -> None:
    import io
    import sys

    def _capture_call(func: Callable, args: tuple, kwargs: dict) -> tuple[str, Any]:
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        old_out = sys.stdout
        old_err = sys.stderr
        sys.stdout = captured_out
        sys.stderr = captured_err
        result = None
        err_msg = ""
        try:
            result = func(*args, **kwargs)
        except Exception:
            import traceback
            err_msg = traceback.format_exc()
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        stdout_val = captured_out.getvalue()
        return stdout_val, (err_msg, result)

    worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(worker_loop)

    async def _do_invoke(agent: AgenticObject, prompt: str) -> Any:
        return await agent.invoke_agent(
            prompt=prompt,
            hooks={
                "on_tool_call": [lambda ctx: print(_c("TOOL", ctx.get("tool_name", "?"))) if state.output_flags.get("TOOL", True) else None],
                "before_tool_execution": [_make_bte(state)],
                "after_tool_execution": [_make_ate(state)],
                "after_message_append": [_make_mappend(state)],
                "on_invoke_complete": [_make_done(state)],
            } if not state.first_invoke else {
                "before_send_to_chatbot": [lambda runner, ctx: _before_llm(runner, ctx, state)],
                "on_tool_call": [lambda ctx: print(_c("TOOL", ctx.get("tool_name", "?"))) if state.output_flags.get("TOOL", True) else None],
                "before_tool_execution": [_make_bte(state)],
                "after_tool_execution": [_make_ate(state)],
                "after_message_append": [_make_mappend(state)],
                "on_invoke_complete": [_make_done(state)],
            },
            persistent_thread_id=state.thread_id,
        )

    async def _run_loop() -> None:
        def _blocking_get():
            return message_queue.get()

        while True:
            msg = await asyncio.get_event_loop().run_in_executor(None, _blocking_get)
            if msg is None:
                break  # None = shutdown sentinel, exit loop

            if isinstance(msg, _FuncCallMsg):
                func = state._func_registry.get(msg.fname)
                stdout_val = ""
                err_msg = ""
                result = None
                if func is not None:
                    try:
                        stdout_val, (err_msg, result) = func(*msg.fargs, **msg.fkwargs)
                    except Exception:
                        import traceback
                        err_msg = traceback.format_exc()
                    state.func_result_queue.put((msg.fname, stdout_val, err_msg, result))
                else:
                    state.func_result_queue.put((msg.fname, "", f"Unknown function: {msg.fname}", None))

            elif isinstance(msg, _PromptMsg):
                prompt = msg.text.strip()
                if not prompt:
                    result_queue.put(None)
                    continue
                agent = agents[state.foreground_agent_name]
                try:
                    result = await _do_invoke(agent, prompt)
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    for line in tb.splitlines():
                        print(_c("ERROR", line))
                    result = None
                state.first_invoke = False
                result_queue.put(result)

    try:
        worker_loop.run_until_complete(_run_loop())
    finally:
        worker_loop.close()


async def _main() -> None:
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=None, help="Fully qualified class path")
    args = parser.parse_args()

    agents: dict[str, AgenticObject] = {}
    if args.agent:
        module_path, class_name = args.agent.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        agents["default"] = getattr(mod, class_name)()
    else:
        agents["default"] = _DemoAgent()

    message_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue[Any] = queue.Queue()
    state = _ShellState()
    state.code_globals = _make_code_globals(agents, state)
    state._func_seen = set()
    state._func_registry = {}
    state.code_queue = queue.Queue()
    state.code_result_queue = queue.Queue()

    worker = threading.Thread(target=_peteos_worker, args=(agents, message_queue, result_queue, state), daemon=True)
    worker.start()

    shell = PSH(agents=agents, message_queue=message_queue, result_queue=result_queue, state=state)

    rc_candidates = [
        os.path.join(os.getcwd(), ".pshrc"),
        os.path.expanduser("~/.pshrc"),
    ]
    for rc_path in rc_candidates:
        if os.path.isfile(rc_path):
            print(_c("SHELL", f"Loading rc from {rc_path}"))
            with open(rc_path) as f:
                rc_content = f.read()
            shell._run_code(rc_content)
            break

    await shell.run_shell()

    message_queue.put(None)
    worker.join(timeout=5.0)


if __name__ == "__main__":
    asyncio.run(_main())
