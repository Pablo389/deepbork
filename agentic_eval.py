from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from main import (
    DEFAULT_DATA_DIR,
    build_messages,
    extract_code,
    generate_text,
    load_alpaca,
    resolve_api_key,
    resolve_model,
)
from evaluate import DEFAULT_MODAL_APP, evaluate_local
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


DEFAULT_PHASE1_REPAIR_RULES = Path("repair_rules/triton_phase1_rules.json")
DEFAULT_PHASE2_REPAIR_RULES = Path("repair_rules/triton_phase2_rules.json")


REPAIR_SYSTEM_PROMPT = """You are repairing a generated Triton Python module.

The previous generated code failed TritonBench-T evaluation.

Phase 1 concatenates the generated Python module with the golden TritonBench-T test driver and executes the resulting file. Phase 2 re-runs surviving Phase 1 modules and compares their outputs with the golden implementation.

Your task:
- Return a complete corrected Python module.
- Preserve the wrapper function name and signature required by the instruction.
- You may rewrite the implementation substantially if the previous structure is invalid.
- Fix the runtime, import, syntax, Triton compilation, or CUDA error shown below.
- If the error is caused by an unsupported Triton language function, replace it with Triton-supported operations.
- Do not include markdown.
- Do not include explanations.
- Do not include tests or example calls.
- Only output Python code.
"""


def parse_args() -> argparse.Namespace:
    provider_default = os.environ.get("LLM_PROVIDER", "modal-vllm")
    model_default = resolve_model(provider_default, None)
    parser = argparse.ArgumentParser(
        description="Run a simple local generate/repair loop against Modal evaluation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing TritonBench-T prompt and metadata files.",
    )
    parser.add_argument(
        "--provider",
        choices=["modal-vllm", "openai"],
        default=provider_default,
        help="LLM provider. Both providers use the OpenAI Python client.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("DEFAULT_ENDPOINT"),
        help="Base URL for modal-vllm. Ignored when --provider openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model name. Current provider default: {model_default}.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override provider API key. Defaults to OPENAI_API_KEY or VLLM_API_KEY.",
    )
    parser.add_argument(
        "--dataset",
        choices=["simp", "comp"],
        default="simp",
        help="TritonBench Alpaca dataset variant.",
    )
    parser.add_argument(
        "--ops",
        required=True,
        help="One TritonBench-T operator filename stem, e.g. div or tanh.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum total evaluation attempts, including the first generation.",
    )
    parser.add_argument(
        "--target-phase",
        choices=["phase1", "phase2"],
        default="phase1",
        help="Stop after Phase 1, or require Phase 1 and Phase 2 to pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agentic_eval"),
        help="Local directory for attempt artifacts.",
    )
    parser.add_argument(
        "--eval-output-prefix",
        default="results/phase1/agentic",
        help="Modal volume prefix for per-attempt evaluation artifacts.",
    )
    parser.add_argument(
        "--modal-app",
        type=Path,
        default=DEFAULT_MODAL_APP,
        help="Path to the Modal evaluation app file.",
    )
    parser.add_argument(
        "--modal-bin",
        default="modal",
        help="Modal executable to invoke.",
    )
    parser.add_argument(
        "--repair-rules",
        type=Path,
        default=DEFAULT_PHASE1_REPAIR_RULES,
        help="JSON file containing curated Triton Phase 1 repair rules.",
    )
    parser.add_argument(
        "--phase2-repair-rules",
        type=Path,
        default=DEFAULT_PHASE2_REPAIR_RULES,
        help="JSON file containing curated Triton Phase 2 repair rules.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per model call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for initial generation and repairs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds per model request.",
    )
    return parser.parse_args()


def select_single_item(data_dir: Path, dataset: str, ops: str) -> tuple[str, dict]:
    requested_files = parse_ops(ops)
    if len(requested_files) != 1:
        raise ValueError("--ops must name exactly one operator for agentic evaluation")

    metadata = load_metadata(data_dir / DEFAULT_METADATA_FILE)
    items = load_alpaca(data_dir, dataset)
    selected = select_items_by_ops(items, metadata, requested_files)
    return requested_files[0], selected[0]


def benchmark_text(item: dict) -> str:
    instr = item["instruction"]
    inp = item.get("input", "") or ""
    return instr if not inp else f"{instr}\n\n{inp}"


def build_repair_messages(
    item: dict,
    previous_predict: str,
    evaluation_result: dict,
    attempt: int,
    max_attempts: int,
    repair_rules: list[dict],
    failed_phase: str,
) -> list[dict]:
    rules = format_repair_rules(
        match_repair_rules(
            previous_predict=previous_predict,
            evaluation_result=evaluation_result,
            repair_rules=repair_rules,
        )
    )
    user = f"""Original benchmark instruction:
{benchmark_text(item)}

Previous generated code:
{previous_predict}

Relevant repair rules:
{rules}

Failed phase:
{failed_phase}

Evaluation stdout:
{evaluation_result.get("stdout_tail", "")}

Evaluation stderr:
{evaluation_result.get("stderr_tail", "")}

Repair attempt:
{attempt + 1} of {max_attempts}
"""
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def load_repair_rules(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"repair rules file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"repair rules file must contain a JSON list: {path}")
    return data


def match_repair_rules(
    previous_predict: str,
    evaluation_result: dict,
    repair_rules: list[dict],
) -> list[dict]:
    evaluation_output = (evaluation_result.get("stdout_tail", "") or "") + "\n" + (
        evaluation_result.get("stderr_tail", "") or ""
    )
    sources = {
        "previous_predict": previous_predict,
        "phase1_output": evaluation_output,
        "phase2_output": evaluation_output,
        "evaluation_output": evaluation_output,
    }
    matched = []
    for rule in repair_rules:
        patterns = rule.get("match", [])
        if any(rule_pattern_matches(pattern, sources) for pattern in patterns):
            matched.append(rule)
    return matched


def rule_pattern_matches(pattern: dict, sources: dict[str, str]) -> bool:
    source = sources.get(pattern.get("source", ""), "")
    if "contains" in pattern:
        return pattern["contains"] in source
    if "regex" in pattern:
        return re.search(pattern["regex"], source, flags=re.DOTALL) is not None
    return False


def format_repair_rules(rules: list[dict]) -> str:
    if not rules:
        return "- No curated rule matched. Use the traceback line number and error message to make the smallest valid Triton fix."

    blocks = []
    for rule in rules:
        lines = [
            f"- [{rule.get('id', 'unnamed_rule')}]",
            f"  Problem: {rule.get('problem', '')}",
            f"  Fix: {rule.get('fix', '')}",
        ]
        avoid = rule.get("avoid") or []
        if avoid:
            lines.append("  Avoid:")
            lines.extend(f"  - {item}" for item in avoid)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def generate_code(
    provider: str,
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    raw = generate_text(
        provider=provider,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return extract_code(raw)


def write_single_prediction(path: Path, instruction: str, predict: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"instruction": instruction, "predict": predict}
    path.write_text(json.dumps(record) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def is_passed(summary: dict, op_file: str, target_phase: str) -> bool:
    if target_phase == "phase2":
        return op_file in set(summary.get("phase2_passed_files", []))
    return op_file in set(summary.get("phase1_passed_files", summary.get("passed_files", [])))


def failed_phase(summary: dict, target_phase: str) -> str:
    if summary.get("failed_phase"):
        return summary["failed_phase"]
    if target_phase == "phase2" and summary.get("phase2_exec_acc", {}).get("failed", 0):
        return "phase2"
    return "phase1"


def validate_generation_config(provider: str, endpoint: str, api_key: str) -> None:
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT or pass --endpoint.")
    if provider == "openai" and not api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key.")


def solve_item(args: argparse.Namespace) -> dict:
    op_file, item = select_single_item(args.data_dir, args.dataset, args.ops)
    op_stem = op_file.removesuffix(".py")

    api_key = resolve_api_key(args.provider, args.api_key)
    model = resolve_model(args.provider, args.model)
    validate_generation_config(args.provider, args.endpoint, api_key)
    phase1_repair_rules = load_repair_rules(args.repair_rules)
    phase2_repair_rules = load_repair_rules(args.phase2_repair_rules)

    op_dir = args.output_dir / op_stem
    op_dir.mkdir(parents=True, exist_ok=True)

    print(f"agentic {args.target_phase} target: {op_file}", flush=True)
    print(f"generating initial candidate with {args.provider}/{model}", flush=True)
    predict = generate_code(
        provider=args.provider,
        endpoint=args.endpoint,
        api_key=api_key,
        model=model,
        messages=build_messages(item),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    last_summary = None
    for attempt in range(1, args.max_attempts + 1):
        attempt_dir = op_dir / f"attempt_{attempt:03d}"
        prediction_path = attempt_dir / "predictions.jsonl"
        write_single_prediction(prediction_path, item["instruction"], predict)
        (attempt_dir / "predict.py").write_text(predict)

        output_subdir = f"{args.eval_output_prefix}/{op_stem}/attempt_{attempt:03d}"
        print(
            f"{args.target_phase} attempt {attempt}/{args.max_attempts}: {output_subdir}",
            flush=True,
        )
        summary = evaluate_local(
            predictions=prediction_path,
            output_subdir=output_subdir,
            modal_app=args.modal_app,
            modal_bin=args.modal_bin,
            target_phase="all" if args.target_phase == "phase2" else "phase1",
        )
        last_summary = summary
        write_json(attempt_dir / "evaluation_summary.json", summary)
        write_json(attempt_dir / "phase1_summary.json", summary)
        metrics = summary.get("phase2_exec_acc" if args.target_phase == "phase2" else "phase1_call_acc", {})
        print(
            f"{args.target_phase} result: "
            f"{metrics.get('passed', 0)}/{summary.get('total_predictions', 0)} "
            f"passed ({metrics.get('rate', 0)}%)",
            flush=True,
        )

        if is_passed(summary, op_file, args.target_phase):
            result = {
                "passed": True,
                "target_phase": args.target_phase,
                "op_file": op_file,
                "attempts": attempt,
                "final_predict_path": str(attempt_dir / "predict.py"),
                "last_evaluation_result": summary,
            }
            write_json(op_dir / "result.json", result)
            return result

        if attempt == args.max_attempts:
            break

        failure_tail = summary.get("stderr_tail") or summary.get("stdout_tail") or ""
        if failure_tail:
            print("evaluation failure tail:\n" + failure_tail[-1200:], flush=True)

        current_failed_phase = failed_phase(summary, args.target_phase)
        current_repair_rules = phase2_repair_rules if current_failed_phase == "phase2" else phase1_repair_rules
        repair_messages = build_repair_messages(
            item=item,
            previous_predict=predict,
            evaluation_result=summary,
            attempt=attempt,
            max_attempts=args.max_attempts,
            repair_rules=current_repair_rules,
            failed_phase=current_failed_phase,
        )
        (attempt_dir / "repair_prompt.json").write_text(json.dumps(repair_messages, indent=2) + "\n")
        print(f"repairing candidate from {current_failed_phase} output", flush=True)
        predict = generate_code(
            provider=args.provider,
            endpoint=args.endpoint,
            api_key=api_key,
            model=model,
            messages=repair_messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )

    result = {
        "passed": False,
        "target_phase": args.target_phase,
        "op_file": op_file,
        "attempts": args.max_attempts,
        "final_predict_path": str(op_dir / f"attempt_{args.max_attempts:03d}" / "predict.py"),
        "last_evaluation_result": last_summary,
    }
    write_json(op_dir / "result.json", result)
    return result


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    result = solve_item(args)
    print(f"\n=== Agentic {args.target_phase} result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
