from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_METADATA_FILE = "TritonBench_T_v1.jsonl"


def load_metadata(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")

    text = path.read_text().strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metadata JSON on line {line_no}: {path}") from exc
    return records


def normalize_op_file(op: str) -> str:
    op = op.strip()
    if not op:
        raise ValueError("empty operator name in --ops")
    if "/" in op or "\\" in op:
        raise ValueError(f"--ops expects bare operator names, got {op!r}")
    return op.removesuffix(".py") + ".py"


def parse_ops(ops: str) -> list[str]:
    if not ops:
        return []
    seen = set()
    normalized = []
    for op in ops.split(","):
        file_name = normalize_op_file(op)
        if file_name not in seen:
            normalized.append(file_name)
            seen.add(file_name)
    return normalized


def extract_functional_description(instruction: str) -> str:
    pattern = r"Functional Description:\s*(.*?)\s*Wrapper Entry Information:"
    match = re.search(pattern, instruction, flags=re.DOTALL)
    if not match:
        raise ValueError("instruction is missing Functional Description block")
    return " ".join(match.group(1).split())


def select_items_by_ops(
    items: list[dict],
    metadata: list[dict],
    requested_files: list[str],
) -> list[dict]:
    metadata_by_file = {record.get("file"): record for record in metadata}
    missing = [file_name for file_name in requested_files if file_name not in metadata_by_file]
    if missing:
        available = ", ".join(sorted(k.removesuffix(".py") for k in metadata_by_file if k)[:20])
        raise ValueError(
            f"unknown --ops target(s): {', '.join(missing)}. "
            f"Examples of available ops: {available}"
        )

    items_by_description: dict[str, list[dict]] = {}
    for item in items:
        description = extract_functional_description(item["instruction"])
        items_by_description.setdefault(description, []).append(item)

    selected = []
    for file_name in requested_files:
        description = " ".join(metadata_by_file[file_name]["description"].split())
        matches = items_by_description.get(description, [])
        if not matches:
            raise ValueError(
                f"metadata target {file_name} has no matching Alpaca prompt "
                "by Functional Description"
            )
        if len(matches) > 1:
            raise ValueError(
                f"metadata target {file_name} matched {len(matches)} Alpaca prompts; "
                "expected exactly one"
            )
        selected.append(matches[0])

    return selected
