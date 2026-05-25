from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tritonbench_helpers import (
    DEFAULT_METADATA_FILE,
    load_metadata,
    parse_ops,
    select_items_by_ops,
)


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATASET = "simp"
DEFAULT_OUTPUT_PATH = Path("outputs/predictions.jsonl")
DEFAULT_TIMEOUT = 600

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
    provider_default = os.environ.get("LLM_PROVIDER", "modal-vllm")
    model_default = resolve_model(provider_default, None)
    parser = argparse.ArgumentParser(
        description="Generate predictions.jsonl with an OpenAI-compatible LLM endpoint."
    )
    parser.add_argument(
        "--provider",
        choices=["modal-vllm", "openai"],
        default=provider_default,
        help="LLM provider. Both providers use the OpenAI Python client.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name. Current provider default: {model_default}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate the first N items. Use 0 for all items. Defaults to 3.",
    )
    parser.add_argument(
        "--ops",
        default="",
        help=(
            "Comma-separated TritonBench-T operator filenames without the .py "
            "extension, e.g. tanh,sqrt,fused_bmm_rmsnorm_gelu_dropout_sub."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
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
        default=DEFAULT_TIMEOUT,
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


def openai_base_url(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def generate_text(
    provider: str,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT.")

    from openai import OpenAI

    if provider == "modal-vllm":
        client = OpenAI(
            api_key=api_key,
            base_url=openai_base_url(endpoint),
            timeout=timeout,
        )
    elif provider == "openai":
        client = OpenAI(api_key=api_key, timeout=timeout)
    else:
        raise ValueError(f"unknown provider {provider!r}")

    resolved_model = resolve_model(provider, model)
    completion = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
    )
    message = completion.choices[0].message
    return (
        message.content
        or getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )


def generate_predictions(
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    limit: int | None = None,
    ops: str = "",
) -> Path:
    endpoint = resolve_endpoint(provider)
    resolved_api_key = resolve_api_key(provider)
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT.")
    if provider == "openai" and not resolved_api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY.")

    metadata_path = DEFAULT_DATA_DIR / DEFAULT_METADATA_FILE
    requested_files = parse_ops(ops)
    items = load_alpaca(DEFAULT_DATA_DIR, DEFAULT_DATASET)

    if requested_files and limit is not None:
        print(
            "warning: both --limit and --ops were provided; ignoring --ops and using --limit",
            flush=True,
        )
        requested_files = []

    if requested_files:
        metadata = load_metadata(metadata_path)
        items = select_items_by_ops(items, metadata, requested_files)
        print(
            "selected ops: " + ", ".join(file_name.removesuffix(".py") for file_name in requested_files),
            flush=True,
        )
    elif limit is None:
        items = items[:3]
    elif limit:
        items = items[:limit]

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target = endpoint.rstrip("/") if provider == "modal-vllm" else "OpenAI API"
    print(
        f"generating {len(items)} predictions sequentially with {provider} ({target})",
        flush=True,
    )

    with DEFAULT_OUTPUT_PATH.open("w") as f:
        for index, item in enumerate(items, start=1):
            try:
                raw = generate_text(
                    provider=provider,
                    endpoint=endpoint,
                    api_key=resolved_api_key,
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

    return DEFAULT_OUTPUT_PATH


def resolve_api_key(provider: str, override: str | None = None) -> str:
    if override is not None:
        return override
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY", "")
    if provider == "modal-vllm":
        return os.environ.get("VLLM_API_KEY", "EMPTY")
    raise ValueError(f"unknown provider {provider!r}")


def resolve_endpoint(provider: str) -> str:
    if provider == "modal-vllm":
        return os.environ.get("DEFAULT_ENDPOINT", "")
    if provider == "openai":
        return ""
    raise ValueError(f"unknown provider {provider!r}")


def resolve_model(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "modal-vllm":
        return os.environ.get("VLLM_MODEL", "llm")
    raise ValueError(f"unknown provider {provider!r}")


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    output_path = generate_predictions(
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        limit=args.limit,
        ops=args.ops,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
