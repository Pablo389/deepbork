# Deepbork Agentic Flow

```mermaid
flowchart TD
    A[Select one TritonBench-T op<br/>agentic_eval.py --ops] --> B[Load prompt + metadata<br/>data/ + tritonbench_helpers.py]
    B --> C[Generate initial candidate<br/>main.py prompt helpers + LLM]
    C --> D[Write attempt artifacts<br/>outputs/agentic_eval/op/attempt_N/]
    D --> E[One-line predictions.jsonl<br/>instruction + predict]

    E --> F[evaluate_local target stage<br/>evaluate.py]
    F --> G{Target stage?}

    G -->|phase1| H[Modal Phase 1<br/>call accuracy]
    G -->|all| H

    H --> H1{Phase 1 passed?}
    H1 -->|no| X[failed_stage = phase1]
    H1 -->|yes| I[call_acc survivors<br/>Modal volume call_acc/]

    I --> J{Need full pipeline?}
    J -->|no target=phase1| P[Accept candidate]
    J -->|yes target=all| K[Modal Phase 2<br/>execution accuracy]

    K --> K1{Phase 2 passed?}
    K1 -->|no| Y[failed_stage = phase2]
    K1 -->|yes| L[Phase 2 survivors<br/>pruned call_acc/]

    L --> N[Modal Phase 3<br/>efficiency benchmark]

    N --> N1{Phase 3 passed?}
    N1 -->|yes| P
    N1 -->|no| Z[failed_stage = phase3]

    P --> Q[Write final result<br/>outputs/agentic_eval/op/result.json]

    X --> R[Collect stdout/stderr tails<br/>evaluation_summary.json]
    Y --> R
    Z --> R

    R --> T[Build repair prompt<br/>instruction + previous code + failed_stage + normalized context]
    T --> U[Generate repaired candidate]
    U --> V{Max attempts reached?}
    V -->|no| D
    V -->|yes| W[Mark failed<br/>write result.json]
```

## Stage Mapping

```text
agentic_eval.py --target-stage phase1
  -> evaluate.py evaluate_local phase1
  -> Modal Phase 1

agentic_eval.py --target-stage all
  -> evaluate.py evaluate_local all
  -> Modal Phase 1 -> Phase 2 -> Phase 3
```

Every repaired candidate restarts evaluation from Phase 1. A repair motivated
by Phase 3 can still break imports, Triton compilation, wrapper semantics, or
Phase 2 correctness, so previous phase results are not reused after code
changes.

## Artifact Locations

```text
Local attempt artifacts:
outputs/agentic_eval/<op>/attempt_N/
  predictions.jsonl
  predict.py
  evaluation_summary.json
  repair_prompt.json

Local final result:
outputs/agentic_eval/<op>/result.json

Modal attempt artifacts:
results/eval/agentic/<op>/attempt_N/
  call_acc/
  perf_results/
  logs/
```
