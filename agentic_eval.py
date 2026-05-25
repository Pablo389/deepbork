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
    resolve_endpoint,
    resolve_model,
)
from evaluate import evaluate_through
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
DEFAULT_PHASE3_REPAIR_RULES = Path("repair_rules/triton_phase3_rules.json")
DEFAULT_AGENTIC_DATA_DIR = DEFAULT_DATA_DIR
DEFAULT_AGENTIC_DATASET = "simp"
DEFAULT_AGENTIC_OUTPUT_DIR = Path("outputs/agentic_eval")
DEFAULT_EVAL_OUTPUT_PREFIX = "results/eval/agentic"
DEFAULT_TIMEOUT = 600
STAGE_ORDER = ("phase1", "phase2", "phase3")


REPAIR_SYSTEM_PROMPT = """You are repairing a generated Triton Python module.

The previous generated code failed TritonBench-T evaluation.

Phase 1 concatenates the generated Python module with the golden TritonBench-T test driver and executes the resulting file. Phase 2 re-runs surviving Phase 1 modules and compares their outputs with the golden implementation. Phase 3 benchmarks Phase 2 survivors with TritonBench-T performance scripts.

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
        "--target-stage",
        choices=STAGE_ORDER,
        default="phase1",
        help="Evaluation stage that each candidate must pass through before it is accepted.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum generated tokens per model call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for initial generation and repairs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result JSON instead of the compact terminal summary.",
    )
    return parser.parse_args()


def select_single_item(ops: str) -> tuple[str, dict]:
    requested_files = parse_ops(ops)
    if len(requested_files) != 1:
        raise ValueError("--ops must name exactly one operator for agentic evaluation")

    metadata = load_metadata(DEFAULT_AGENTIC_DATA_DIR / DEFAULT_METADATA_FILE)
    items = load_alpaca(DEFAULT_AGENTIC_DATA_DIR, DEFAULT_AGENTIC_DATASET)
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
    failed_stage: str,
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

Failed evaluation stage:
{failed_stage}

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
        "phase3_output": evaluation_output,
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
) -> str:
    raw = generate_text(
        provider=provider,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=DEFAULT_TIMEOUT,
    )
    return extract_code(raw)


def write_single_prediction(path: Path, instruction: str, predict: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"instruction": instruction, "predict": predict}
    path.write_text(json.dumps(record) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def is_passed(summary: dict, op_file: str, target_stage: str) -> bool:
    if target_stage == "phase3":
        return op_file in set(summary.get("phase3_passed_files", []))
    if target_stage == "phase2":
        return op_file in set(summary.get("phase2_passed_files", []))
    return op_file in set(summary.get("phase1_passed_files", summary.get("passed_files", [])))


def passed_through(summary: dict, op_file: str) -> str | None:
    for stage in reversed(STAGE_ORDER):
        if is_passed(summary, op_file, stage):
            return stage
    return None


def failed_stage(summary: dict, target_stage: str) -> str:
    if summary.get("failed_phase"):
        return summary["failed_phase"]
    if target_stage == "phase3" and summary.get("phase3_efficiency", {}).get("status") == "failed":
        return "phase3"
    if target_stage == "phase2" and summary.get("phase2_exec_acc", {}).get("failed", 0):
        return "phase2"
    return "phase1"


def phase_line(summary: dict, stage: str) -> str:
    if stage == "phase1":
        metrics = summary.get("phase1_call_acc", {})
        passed = metrics.get("passed", 0)
        failed = metrics.get("failed", 0)
        rate = metrics.get("rate", 0)
        return f"phase1: {passed} passed, {failed} failed ({rate}%)"
    if stage == "phase2":
        metrics = summary.get("phase2_exec_acc", {})
        passed = metrics.get("passed", 0)
        failed = metrics.get("failed", 0)
        rate = metrics.get("rate", 0)
        return f"phase2: {passed} passed, {failed} failed ({rate}%)"
    phase3 = summary.get("phase3_efficiency", {})
    status = phase3.get("status", "not-run")
    speedup = phase3.get("speedup_vs_pytorch")
    return f"phase3: {status}, speedup={speedup}"


def print_attempt_summary(summary: dict, op_file: str, target_stage: str) -> None:
    stages = STAGE_ORDER[: STAGE_ORDER.index(target_stage) + 1]
    lines = [phase_line(summary, stage) for stage in stages]
    print("evaluation: " + " | ".join(lines), flush=True)
    print(f"passed through: {passed_through(summary, op_file) or 'none'}", flush=True)


def compact_result(result: dict) -> dict:
    summary = result.get("last_evaluation_result") or {}
    return {
        "passed": result.get("passed"),
        "target_stage": result.get("target_stage"),
        "passed_through": result.get("passed_through"),
        "op_file": result.get("op_file"),
        "attempts": result.get("attempts"),
        "final_predict_path": result.get("final_predict_path"),
        "result_path": result.get("result_path"),
        "artifacts_volume": summary.get("artifacts_volume"),
        "artifacts_subdir": summary.get("artifacts_subdir"),
        "call_acc_dir": summary.get("call_acc_dir"),
        "perf_results_dir": summary.get("perf_results_dir"),
        "phase1": summary.get("phase1_call_acc"),
        "phase2": summary.get("phase2_exec_acc"),
        "phase3": summary.get("phase3_efficiency"),
        "failed_phase": summary.get("failed_phase"),
    }


def validate_generation_config(provider: str, endpoint: str, api_key: str) -> None:
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT.")
    if provider == "openai" and not api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY.")


def solve_item(args: argparse.Namespace) -> dict:
    op_file, item = select_single_item(args.ops)
    op_stem = op_file.removesuffix(".py")

    endpoint = resolve_endpoint(args.provider)
    api_key = resolve_api_key(args.provider)
    model = resolve_model(args.provider, args.model)
    validate_generation_config(args.provider, endpoint, api_key)
    phase1_repair_rules = load_repair_rules(DEFAULT_PHASE1_REPAIR_RULES)
    phase2_repair_rules = load_repair_rules(DEFAULT_PHASE2_REPAIR_RULES)
    phase3_repair_rules = load_repair_rules(DEFAULT_PHASE3_REPAIR_RULES)

    op_dir = DEFAULT_AGENTIC_OUTPUT_DIR / op_stem
    op_dir.mkdir(parents=True, exist_ok=True)

    print(f"agentic target stage: {args.target_stage}, op: {op_file}", flush=True)
    print(f"generating initial candidate with {args.provider}/{model}", flush=True)
    predict = generate_code(
        provider=args.provider,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        messages=build_messages(item),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    last_summary = None
    for attempt in range(1, args.max_attempts + 1):
        attempt_dir = op_dir / f"attempt_{attempt:03d}"
        prediction_path = attempt_dir / "predictions.jsonl"
        write_single_prediction(prediction_path, item["instruction"], predict)
        (attempt_dir / "predict.py").write_text(predict)

        output_subdir = f"{DEFAULT_EVAL_OUTPUT_PREFIX}/{op_stem}/attempt_{attempt:03d}"
        print(
            f"attempt {attempt}/{args.max_attempts}: evaluate through {args.target_stage} -> {output_subdir}",
            flush=True,
        )
        summary = evaluate_through(
            predictions=prediction_path,
            output_subdir=output_subdir,
            target_stage=args.target_stage,
        )
        last_summary = summary
        write_json(attempt_dir / "evaluation_summary.json", summary)
        print_attempt_summary(summary, op_file, args.target_stage)

        if is_passed(summary, op_file, args.target_stage):
            result = {
                "passed": True,
                "target_stage": args.target_stage,
                "passed_through": args.target_stage,
                "op_file": op_file,
                "attempts": attempt,
                "final_predict_path": str(attempt_dir / "predict.py"),
                "last_evaluation_result": summary,
            }
            result["result_path"] = str(op_dir / "result.json")
            write_json(op_dir / "result.json", result)
            return result

        if attempt == args.max_attempts:
            break

        failure_tail = summary.get("stderr_tail") or summary.get("stdout_tail") or ""
        if failure_tail:
            print("evaluation failure tail:\n" + failure_tail[-1200:], flush=True)

        current_failed_stage = failed_stage(summary, args.target_stage)
        repair_rules_by_stage = {
            "phase1": phase1_repair_rules,
            "phase2": phase2_repair_rules,
            "phase3": phase3_repair_rules,
        }
        current_repair_rules = repair_rules_by_stage.get(current_failed_stage, phase1_repair_rules)
        repair_messages = build_repair_messages(
            item=item,
            previous_predict=predict,
            evaluation_result=summary,
            attempt=attempt,
            max_attempts=args.max_attempts,
            repair_rules=current_repair_rules,
            failed_stage=current_failed_stage,
        )
        (attempt_dir / "repair_prompt.json").write_text(json.dumps(repair_messages, indent=2) + "\n")
        print(f"repairing candidate from {current_failed_stage} output", flush=True)
        predict = generate_code(
            provider=args.provider,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            messages=repair_messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    result = {
        "passed": False,
        "target_stage": args.target_stage,
        "passed_through": passed_through(last_summary or {}, op_file),
        "op_file": op_file,
        "attempts": args.max_attempts,
        "final_predict_path": str(op_dir / f"attempt_{args.max_attempts:03d}" / "predict.py"),
        "last_evaluation_result": last_summary,
    }
    result["result_path"] = str(op_dir / "result.json")
    write_json(op_dir / "result.json", result)
    return result


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    result = solve_item(args)
    print(f"\n=== Agentic result: {'passed' if result.get('passed') else 'failed'} ===")
    print(json.dumps(result if args.json else compact_result(result), indent=2))


if __name__ == "__main__":
    main()
