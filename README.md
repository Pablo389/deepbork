# deepbork

Local prediction-generation repo for Deepbork.

This repo only creates a `predictions.jsonl` file. It does not build a Modal
image, run Modal, execute TritonBench evaluation, or upload artifacts anywhere.

The current flow is:

1. prepare a local checkout of upstream TritonBench
2. read the TritonBench-T Alpaca dataset
3. call the local `llms_xgrammar` generator sequentially on one local/self-hosted GPU
4. write one JSON line per generated operator

## Contract

The local LLM backend receives the same message shape used by the existing
generation contract:

```python
[
    {"role": "system", "content": PROMPT_HEADER},
    {"role": "user", "content": instruction_or_instruction_plus_input},
]
```

The local LLM backend returns raw generated text. `deepbork` then applies the
same code-fence cleanup logic used by the current benchmark harness and writes
JSONL records:

```json
{"instruction": "...", "predict": "..."}
```

The resulting file is the only artifact this repo is responsible for.

## Setup

Clone `deepbork` and `llms-xgrammar` into the same parent directory:

```bash
git clone https://github.com/Pablo389/deepbork
git clone https://github.com/Pablo389/llms-xgrammar
```

Expected layout:

```text
workspace/
  deepbork/
  llms-xgrammar/
```

Install PyTorch for your machine first. Use the official PyTorch selector for
CPU, CUDA, or your target backend:

```text
https://pytorch.org/get-started/locally/
```

Then install the Python dependencies used by the local generator:

```bash
cd deepbork
python3 -m pip install -r requirements.txt
```

`requirements.txt` does not install `llms-xgrammar`. The script imports it from
the sibling checkout shown above.

## Generate Predictions

Use an existing TritonBench checkout:

```bash
python3 main.py \
  --tritonbench-dir /path/to/TritonBench \
  --dataset simp \
  --limit 1 \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --grammar-name triton_lexical \
  --max-new-tokens 256 \
  --output outputs/predictions.jsonl
```

Or let the script clone TritonBench into `vendor/TritonBench`:

```bash
python3 main.py --limit 1 --max-new-tokens 256
```

The generated file is written to `outputs/predictions.jsonl` by default.

## Scope

In scope:

- clone upstream TritonBench if needed
- read `data/TritonBench_T_<simp|comp>_alpac_v1.json`
- build prompt messages
- call `llms_xgrammar.generate_text(...)`
- write `predictions.jsonl`

Out of scope:

- Modal image build
- TritonBench eval script patching
- benchmark execution
- uploading predictions
- parallel LLM calls
