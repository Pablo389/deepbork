from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


DEFAULT_DATA_DIR = Path("data")

PROMPT_HEADER = (
    "You are an expert in Triton programming, capable of writing Triton kernels "
    "and wrapper functions based on functional descriptions and function "
    "parameters. The wrapper function must fully match the provided function "
    "signature.\n\n"
    "Output a single, self-contained Python module containing: (a) the necessary "
    "imports (torch, triton, triton.language as tl), (b) the Triton kernel(s), "
    "and (c) the wrapper function that the description specifies. Wrap the "
    "entire module in one ```python ... ``` fenced code block. Do NOT include "
    "any test code or example calls — tests will be appended separately."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predictions.jsonl with an OpenAI-compatible LLM endpoint."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing TritonBench_T_<simp|comp>_alpac_v1.json.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("DEFAULT_ENDPOINT"),
        help="Base URL for the deployed OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEFAULT_MODEL", "llm"),
        help="Served model name. modal_vllm exposes this as 'llm' by default.",
    )
    parser.add_argument(
        "--dataset",
        choices=["simp", "comp"],
        default="simp",
        help="TritonBench Alpaca dataset variant.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only generate the first N items. Use 0 for all items.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
        help="Output predictions.jsonl path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per benchmark item.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds per request.",
    )
    return parser.parse_args()


def load_alpaca(data_dir: Path, dataset: str) -> list[dict]:
    path = data_dir / f"TritonBench_T_{dataset}_alpac_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    return json.loads(path.read_text())


def build_messages(item: dict) -> list[dict]:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    user = instr if not inp else f"{instr}\n\n{inp}"
    return [
        {"role": "system", "content": PROMPT_HEADER},
        {"role": "user", "content": user},
    ]


def extract_code(text: str) -> str:
    """Same cleanup contract as TritonBench4Modal/modal_app.py."""
    import re

    s = text.strip()
    m = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


def generate_text(
    endpoint: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    if not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT or pass --endpoint.")

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"endpoint returned HTTP {exc.code}: {error_body}") from exc

    data = json.loads(response_body)
    message = data["choices"][0]["message"]
    return (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )


def generate_predictions(
    data_dir: Path,
    endpoint: str,
    model: str,
    dataset: str,
    output_path: Path,
    max_tokens: int,
    temperature: float,
    timeout: int,
    limit: int = 0,
) -> Path:
    if not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT or pass --endpoint.")

    items = load_alpaca(data_dir, dataset)
    if limit:
        items = items[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"generating {len(items)} predictions sequentially with {endpoint.rstrip('/')}",
        flush=True,
    )

    with output_path.open("w") as f:
        for index, item in enumerate(items, start=1):
            try:
                raw = generate_text(
                    endpoint=endpoint,
                    model=model,
                    messages=build_messages(item),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                )
                code = extract_code(raw)
            except Exception as exc:  # noqa: BLE001
                code = f"# generation failed: {exc}\n"

            record = {"instruction": item["instruction"], "predict": code}
            f.write(json.dumps(record) + "\n")
            print(f"  {index}/{len(items)}", flush=True)

    return output_path


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    output_path = generate_predictions(
        data_dir=args.data_dir,
        endpoint=args.endpoint,
        model=args.model,
        dataset=args.dataset,
        output_path=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        limit=args.limit,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
