from __future__ import annotations

import argparse
import json
import os
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
from evaluate import (
    STAGE_ORDER,
    evaluate_local,
    failed_stage,
    is_passed,
    passed_through,
    phase_result,
    phase_line,
)
from tritonbench_helpers import (
    DEFAULT_METADATA_FILE,
    item_file_pairs,
    load_metadata,
    parse_ops,
    select_items_by_ops,
)


try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


DEFAULT_AGENTIC_DATA_DIR = DEFAULT_DATA_DIR
DEFAULT_AGENTIC_DATASET = "simp"
DEFAULT_AGENTIC_OUTPUT_DIR = Path("outputs/agentic_eval")
DEFAULT_EVAL_OUTPUT_PREFIX = "results/eval/agentic"
DEFAULT_TIMEOUT = 600
AGENTIC_TARGETS = ("phase1", "all")
PASS_STAGE = {
    "phase1": "phase1",
    "all": "phase3",
}


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
        default="",
        help=(
            "Comma-separated TritonBench-T operator filename stems, e.g. div,tanh. "
            "If omitted, --limit selects from the dataset."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run agentic evaluation for the first N dataset items. Use 0 for all items.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum total evaluation attempts, including the first generation.",
    )
    parser.add_argument(
        "--target-stage",
        choices=AGENTIC_TARGETS,
        default="phase1",
        help="Acceptance target: Phase 1 only or the complete Phase 1+2+3 pipeline.",
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


def select_items(args: argparse.Namespace) -> list[tuple[str, dict]]:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0")

    requested_files = parse_ops(args.ops)
    metadata = load_metadata(DEFAULT_AGENTIC_DATA_DIR / DEFAULT_METADATA_FILE)
    items = load_alpaca(DEFAULT_AGENTIC_DATA_DIR, DEFAULT_AGENTIC_DATASET)

    if requested_files and args.limit is not None:
        print(
            "warning: both --limit and --ops were provided; ignoring --ops and using --limit",
            flush=True,
        )
        requested_files = []

    if requested_files:
        selected = select_items_by_ops(items, metadata, requested_files)
        return list(zip(requested_files, selected))

    if args.limit is None:
        items = items[:1]
    elif args.limit:
        items = items[: args.limit]

    return item_file_pairs(items, metadata)


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
    failed_stage: str,
) -> list[dict]:
    user = f"""Original benchmark instruction:
{benchmark_text(item)}

Previous generated code:
{previous_predict}

Failed evaluation stage:
{failed_stage}

Evaluation stdout:
{evaluation_result.get("stdout_tail", "")}

Evaluation stderr:
{evaluation_result.get("stderr_tail", "")}

Passed files:
{json.dumps(evaluation_result.get("passed_files", []))}

Failed files:
{json.dumps(evaluation_result.get("failed_files", []))}

Evaluation artifacts:
{json.dumps(evaluation_result.get("artifacts", {}), indent=2)}

Passed through:
{evaluation_result.get("passed_through")}

Repair attempt:
{attempt + 1} of {max_attempts}
"""
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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


def pass_stage(target_stage: str) -> str:
    return PASS_STAGE[target_stage]


def print_attempt_summary(summary: dict, op_file: str, target_stage: str) -> None:
    target = pass_stage(target_stage)
    stages = STAGE_ORDER[: STAGE_ORDER.index(target) + 1]
    lines = [phase_line(summary, stage) for stage in stages]
    print("evaluation: " + " | ".join(lines), flush=True)
    print(f"passed through: {passed_through(summary, op_file) or 'none'}", flush=True)


def compact_result(result: dict) -> dict:
    summary = result.get("last_evaluation_result") or {}
    artifacts = summary.get("artifacts", {})
    return {
        "passed": result.get("passed"),
        "target_stage": result.get("target_stage"),
        "passed_through": result.get("passed_through"),
        "op_file": result.get("op_file"),
        "attempts": result.get("attempts"),
        "final_predict_path": result.get("final_predict_path"),
        "result_path": result.get("result_path"),
        "artifacts_volume": artifacts.get("artifacts_volume"),
        "artifacts_subdir": artifacts.get("artifacts_subdir"),
        "call_acc_dir": artifacts.get("call_acc_dir"),
        "perf_results_dir": artifacts.get("perf_results_dir"),
        "phase1": (phase_result(summary, "phase1") or {}).get("metrics"),
        "phase2": (phase_result(summary, "phase2") or {}).get("metrics"),
        "phase3": (phase_result(summary, "phase3") or {}).get("metrics"),
        "failed_phase": summary.get("failed_phase"),
    }


def compact_batch_result(result: dict) -> dict:
    return {
        "total": result.get("total", 0),
        "passed": result.get("passed", 0),
        "failed": result.get("failed", 0),
        "result_path": result.get("result_path"),
        "results": [compact_result(item) for item in result.get("results", [])],
    }


def validate_generation_config(provider: str, endpoint: str, api_key: str) -> None:
    if provider == "modal-vllm" and not endpoint:
        raise ValueError("Missing endpoint. Set DEFAULT_ENDPOINT.")
    if provider == "openai" and not api_key:
        raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY.")


def solve_item(args: argparse.Namespace, op_file: str, item: dict) -> dict:
    op_stem = op_file.removesuffix(".py")

    endpoint = resolve_endpoint(args.provider)
    api_key = resolve_api_key(args.provider)
    model = resolve_model(args.provider, args.model)
    validate_generation_config(args.provider, endpoint, api_key)

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
        summary = evaluate_local(
            mode=args.target_stage,
            predictions=prediction_path,
            output_subdir=output_subdir,
        )
        last_summary = summary
        write_json(attempt_dir / "evaluation_summary.json", summary)
        print_attempt_summary(summary, op_file, args.target_stage)

        target_pass_stage = pass_stage(args.target_stage)
        if is_passed(summary, op_file, target_pass_stage):
            result = {
                "passed": True,
                "target_stage": args.target_stage,
                "passed_through": target_pass_stage,
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

        current_failed_stage = failed_stage(summary, pass_stage(args.target_stage))
        failed_result = dict(phase_result(summary, current_failed_stage) or {})
        failed_result["artifacts"] = summary.get("artifacts", failed_result.get("artifacts", {}))
        failed_result["passed_through"] = summary.get("passed_through")
        failure_tail = failed_result.get("stderr_tail") or failed_result.get("stdout_tail") or ""
        if failure_tail:
            print("evaluation failure tail:\n" + failure_tail[-1200:], flush=True)

        repair_messages = build_repair_messages(
            item=item,
            previous_predict=predict,
            evaluation_result=failed_result,
            attempt=attempt,
            max_attempts=args.max_attempts,
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


def solve_batch(args: argparse.Namespace) -> dict:
    selected = select_items(args)
    endpoint = resolve_endpoint(args.provider)
    api_key = resolve_api_key(args.provider)
    validate_generation_config(args.provider, endpoint, api_key)

    print(f"agentic batch size: {len(selected)}", flush=True)
    results = []
    for index, (op_file, item) in enumerate(selected, start=1):
        print(f"\n=== Agentic item {index}/{len(selected)}: {op_file} ===", flush=True)
        results.append(solve_item(args, op_file, item))

    batch_result = {
        "total": len(results),
        "passed": sum(1 for result in results if result.get("passed")),
        "failed": sum(1 for result in results if not result.get("passed")),
        "results": results,
    }
    batch_result["result_path"] = str(DEFAULT_AGENTIC_OUTPUT_DIR / "batch_result.json")
    write_json(DEFAULT_AGENTIC_OUTPUT_DIR / "batch_result.json", batch_result)
    return batch_result


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    args = parse_args()
    result = solve_batch(args)
    print(
        f"\n=== Agentic batch result: {result.get('passed')} passed, {result.get('failed')} failed ==="
    )
    print(json.dumps(result if args.json else compact_batch_result(result), indent=2))


if __name__ == "__main__":
    main()
