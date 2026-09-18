from __future__ import annotations

import json
from dataclasses import dataclass

from peteos import AgenticObject
from peteos.conversation import ContentPart
from peteos.engine import ExecStatus
from peteos.engine import Runner
from peteos.oap.agentic_object import _collect_oap_config
from petekit.text.buffer_manager import BufferManager
from petekit.utils.text_formatters import format_dict_list_for_buffer


@dataclass
class ReflectionCatalog:
    hardcoded: list[dict]
    runtime: list[dict]
    dynamic_functions: bool = False


class SelfReflector(BufferManager, AgenticObject):
    """You maintain two buffers listing every member method you can call on self,
    i.e. for each definition `f(...)` in that list, you can call `self.f(...)` from within python.
    - docs:reflect:sandbox — hardcoded @sandbox and @tool methods. Always present. Parse with json.loads.
    - docs:reflect:dynamic — runtime-defined methods. Only present when the define_functions tool is available.
    Consult these buffers before writing Python code so you know which methods are available on self."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._self_reflection_catalog = ReflectionCatalog(
            hardcoded=self._describe_sandbox_methods(),
            runtime=[],
        )

        # Detect define_functions to know if agent may define runtime methods
        config = _collect_oap_config(self.__class__)
        if config.get("define_functions", False):
            self._self_reflection_catalog.dynamic_functions = True

        # Populate sandbox buffer at construction time with hardcoded @sandbox/@tool methods
        sandbox_text = format_dict_list_for_buffer(self._self_reflection_catalog.hardcoded)
        self.create_buffer("docs:reflect:sandbox", text=sandbox_text)
        if self._self_reflection_catalog.dynamic_functions:
            self.create_buffer("docs:reflect:dynamic", text="[]")
            # TODO: Hook registration: attach_class after_tool_execution for define_function / remove_function to rebuild sandbox_api buffer

    def _self_reflection_on_tool_executed(self, runner: Runner, tc: ContentPart, status: ExecStatus, ok: bool) -> None:
        # Guard: only react to define_function / remove_function — no-op for all other tools
        if tc.name not in ("define_function", "remove_function"):
            return

        if tc.name == "define_function":
            # Diff current sandbox methods against hardcoded set, append delta to runtime catalog
            current = self._describe_sandbox_methods()
            hardcoded_names = {m["name"] for m in self._self_reflection_catalog.hardcoded}
            for item in current:
                if item["name"] not in hardcoded_names:
                    self._self_reflection_catalog.runtime.append(item)

        elif tc.name == "remove_function":
            # Parse function name from tool arguments and remove it from runtime catalog
            args = json.loads(tc.arguments) if tc.arguments else {}
            name = args.get("name")
            if name:
                self._self_reflection_catalog.runtime = [m for m in self._self_reflection_catalog.runtime if m["name"] != name]

        # Refresh dynamic buffer to reflect current runtime catalog
        dynamic_text = format_dict_list_for_buffer(self._self_reflection_catalog.runtime)
        self.create_buffer("docs:reflect:dynamic", text=dynamic_text, overwrite=True)
