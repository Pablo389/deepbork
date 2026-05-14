from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"
DEFAULT_TRITONBENCH_DIR = Path("vendor/TritonBench")

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
        description="Generate predictions.jsonl sequentially with llms_xgrammar."
    )
    parser.add_argument(
        "--tritonbench-dir",
        type=Path,
        default=DEFAULT_TRITONBENCH_DIR,
        help="Path to a TritonBench checkout. Cloned if missing.",
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
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--grammar-name",
        default="triton_lexical",
        help="llms_xgrammar grammar key.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Generation device.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum generated tokens per benchmark item.",
    )
    return parser.parse_args()


def add_sibling_llms_xgrammar_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    workspace_root = repo_root.parent
    candidates = [
        workspace_root / "llms-xgrammar",
        workspace_root / "llms_xgrammar",
    ]
    for candidate in candidates:
        if (candidate / "llms_xgrammar" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return


def load_generate_text():
    add_sibling_llms_xgrammar_to_path()
    try:
        from llms_xgrammar import generate_text
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import llms_xgrammar. Clone "
            "https://github.com/Pablo389/llms-xgrammar next to this repo, or "
            "install it in the current Python environment."
        ) from exc
    return generate_text


def ensure_tritonbench_repo(repo_dir: Path) -> Path:
    if repo_dir.exists():
        print(f"using TritonBench checkout at {repo_dir}", flush=True)
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning TritonBench into {repo_dir}", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", TRITONBENCH_REPO, str(repo_dir)],
        check=True,
    )
    return repo_dir


def load_alpaca(tritonbench_dir: Path, dataset: str) -> list[dict]:
    path = tritonbench_dir / f"data/TritonBench_T_{dataset}_alpac_v1.json"
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


def generate_predictions(
    tritonbench_dir: Path,
    dataset: str,
    output_path: Path,
    model_name: str,
    grammar_name: str,
    device: str,
    max_new_tokens: int,
    limit: int = 0,
) -> Path:
    generate_text = load_generate_text()
    items = load_alpaca(tritonbench_dir, dataset)
    if limit:
        items = items[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"generating {len(items)} predictions sequentially with llms_xgrammar/{model_name}",
        flush=True,
    )

    with output_path.open("w") as f:
        for index, item in enumerate(items, start=1):
            try:
                raw = generate_text(
                    messages=build_messages(item),
                    model_name=model_name,
                    grammar_name=grammar_name,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                code = extract_code(raw)
            except Exception as exc:  # noqa: BLE001
                code = f"# generation failed: {exc}\n"

            record = {"instruction": item["instruction"], "predict": code}
            f.write(json.dumps(record) + "\n")
            print(f"  {index}/{len(items)}", flush=True)

    return output_path


def main() -> None:
    args = parse_args()
    tritonbench_dir = ensure_tritonbench_repo(args.tritonbench_dir)
    output_path = generate_predictions(
        tritonbench_dir=tritonbench_dir,
        dataset=args.dataset,
        output_path=args.output,
        model_name=args.model_name,
        grammar_name=args.grammar_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
