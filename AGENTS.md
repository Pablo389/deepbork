# Repository Guidelines

## Project Structure & Module Organization

Deepbork is a Python pipeline for generating TritonBench-T candidates, evaluating them on Modal, and repairing failures.

- `main.py`: batch code generation; writes `outputs/predictions.jsonl`.
- `evaluate.py`: local CLI wrapper for Modal evaluation phases.
- `modal_eval_app.py`: Modal image, volume, GPU execution, and phase entrypoints.
- `agentic_eval.py`: one-operator generate/evaluate/repair loop.
- `tritonbench_helpers.py`: prompt, metadata, and operator matching helpers.
- `data/`: local TritonBench-T prompt and metadata files.
- `outputs/`: generated local artifacts; treat as runtime output.

## Build, Test, and Development Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run a small generation smoke test:

```bash
python3 main.py --provider modal-vllm --limit 1
```

Generate a specific operator:

```bash
python3 main.py --provider openai --ops tanh
```

Run evaluation through all phases on Modal:

```bash
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl
```

Run the agentic repair loop for one operator:

```bash
python3 agentic_eval.py --provider modal-vllm --ops div --target-stage all
```

Modal evaluation requires `modal setup` and a GPU-capable Modal workspace.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type annotations where useful, and `pathlib.Path` for filesystem paths. Keep CLI defaults as named module constants such as `DEFAULT_OUTPUT_PATH`. Use descriptive snake_case for functions, variables, and JSON field names.

No formatter or linter config is committed. Keep imports tidy, avoid unrelated refactors, and preserve the direct `argparse` CLI style.

## Testing Guidelines

There is no standalone unit test suite yet. Validate changes with the narrowest relevant command:

- generation changes: `python3 main.py --provider ... --limit 1` or `--ops <name>`
- evaluator changes: `python3 evaluate.py --mode phase1 --predictions outputs/predictions.jsonl`
- agentic changes: `python3 agentic_eval.py --ops <name> --max-attempts 1`

## Commit & Pull Request Guidelines

Recent commits use short, direct summaries, for example `Added phase 3: complete evaluation phases`. Keep commit messages concise and behavior-focused.

Pull requests should include a short description, commands run, Modal artifact paths used for verification, and environment assumptions such as `DEFAULT_ENDPOINT`, `OPENAI_API_KEY`, or Modal volume names. Include screenshots only when terminal or Modal UI output is relevant.

## Security & Configuration Tips

Keep secrets out of source. Use `.env` or shell exports for `DEFAULT_ENDPOINT`, `VLLM_API_KEY`, and `OPENAI_API_KEY`. Do not commit generated `outputs/` artifacts unless they are intentionally curated examples.
