"""PSH Simple - a line-by-line shell for agentic objects, no prompt_toolkit."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

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
}


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

    def __init__(self, agent: AgenticObject) -> None:
        super().__init__()
        self._agent = agent
        self._running = False
        self._first_invoke = True
        self._thread_id: str | None = "default"

    def _h(self, ctx: dict) -> None:
        prompt = ctx.get("prompt", "")
        if isinstance(prompt, list):
            prompt = " ".join(str(p) for p in prompt)
        print(_c("INVOKE", f"{prompt!r:.80}", hook="_h(on_invoke)"))

    def _htc(self, ctx: dict) -> None:
        print(_c("TOOL", ctx.get("tool_name", "?")))

    def _bte(self, tc: Any) -> None:
        n = getattr(tc, "name", "?") if tc else "?"
        raw = getattr(tc, "raw_dict", None)
        args_str = raw.get("arguments", "{}") if raw else "{}"
        print(_c("RUN", f"{n}({_fmt_json(args_str)})"))

    def _ate(self, *args: Any, **kwargs: Any) -> None:
        runner = args[0]
        tc = args[1]
        res = args[2]
        ok = args[3]
        n = getattr(tc, "name", "?") if tc else "?"
        res_str = str(res) if res else ""
        if ok:
            print(_c("RET", f"{n} → {_fmt_json(res_str)}"))
        else:
            print(_c("ERR", f"{n}: {_fmt_json(res_str)}"))

    def _mappend(self, runner: Any, msg: Any) -> None:
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

    def _done(self, ctx: dict) -> None:
        r = ctx.get("result")
        if isinstance(r, Error):
            print(_c("ERROR", r.message))
        elif r is not None:
            print(_c("RESULT", repr(r)))
        else:
            print(_c("RESULT", "None"))

    def _before_llm(self, runner: Any, ctx: Any) -> None:
        if not self._first_invoke:
            return
        self._first_invoke = False
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

    def _base_hooks(self, first: bool = False) -> dict[str, list[Callable]]:
        if first:
            return {
                "before_send_to_chatbot": [self._before_llm],
                "on_tool_call": [self._htc],
                "before_tool_execution": [self._bte],
                "after_tool_execution": [self._ate],
                "after_message_append": [self._mappend],
                "on_invoke_complete": [self._done],
            }
        return {
            #"on_invoke": [self._h],  # put this back in to debug the invocation arguments
            "on_tool_call": [self._htc],
            "before_tool_execution": [self._bte],
            "after_tool_execution": [self._ate],
            "after_message_append": [self._mappend],
            "on_invoke_complete": [self._done],
        }

    async def run_shell(self) -> str:
        from prompt_toolkit.shortcuts import prompt
        print(f"{self._TITLE}")
        print("Type /help for commands, /quit to exit.")
        while True:
            try:
                raw = await asyncio.get_running_loop().run_in_executor(None, lambda: prompt("> "))
            except (EOFError, KeyboardInterrupt):
                print("\n[SHELL] EOF — bye")
                break
            raw = raw.strip()
            if not raw:
                continue
            if raw.startswith("/"):
                if self._shell_cmd(raw):
                    break
            else:
                await self._send(raw)
        return "Shell closed."

    async def _send(self, prompt: str) -> None:
        result = await self._agent.invoke_agent(
            prompt=prompt,
            hooks=self._base_hooks(first=self._first_invoke),
            persistent_thread_id=self._thread_id,
        )

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
                "  /help   this message\n"
                "  <text>  send to agent"
                ))
        else:
            print(_c("SHELL", f"Unknown command: /{cmd}"))
        return False

    def _session_cmd(self, args: list[str] | None) -> None:
        if args:
            if args[0].lower() == "none":
                self._thread_id = None
                self._first_invoke = True
                print(_c("SESSION", "Session: none — anonymous (no memory)"))
            else:
                self._thread_id = args[0]
                self._first_invoke = True
                print(_c("SESSION", f"Switched to session: {self._thread_id}"))
        else:
            if self._thread_id:
                print(_c("SESSION", f"Current session: {self._thread_id}"))
            else:
                print(_c("SESSION", "Session: none — anonymous (no memory)"))

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


async def _main() -> None:
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=None, help="Fully qualified class path")
    args = parser.parse_args()

    if args.agent:
        module_path, class_name = args.agent.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        agent = getattr(mod, class_name)()
    else:
        agent = _DemoAgent()

    shell = PSH(agent=agent)
    await shell.run_shell()


if __name__ == "__main__":
    asyncio.run(_main())
