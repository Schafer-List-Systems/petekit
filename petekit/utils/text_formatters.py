from __future__ import annotations

import json


def format_dict_list_for_buffer(records: list[dict]) -> str:
    """Format a list of dicts as a line-by-line JSON array."""
    if not records:
        return "[]"
    json_lines = [json.dumps(r) for r in records]
    return "[\n" + ",\n".join(json_lines) + "\n]"
