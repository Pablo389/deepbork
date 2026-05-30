from __future__ import annotations

from pathlib import Path

from evaluation.model import PHASE_ORDER


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DATASET = "simp"
DEFAULT_PREDICTIONS_PATH = Path("outputs/predictions.jsonl")
DEFAULT_TIMEOUT = 600

DEFAULT_MODAL_APP = Path(__file__).with_name("modal_eval_app.py")
DEFAULT_MODAL_APP_NAME = "deepbork-eval"
DEFAULT_MODAL_BIN = "modal"
DEFAULT_MODAL_VOLUME = "deepbork-data"

EVAL_JSON_START = "DEEPBORK_EVAL_JSON_START"
EVAL_JSON_END = "DEEPBORK_EVAL_JSON_END"

DEFAULT_PHASE1_OUTPUT_SUBDIR = "results/phase1"
DEFAULT_PHASE2_OUTPUT_SUBDIR = "results/phase2"
DEFAULT_PHASE3_OUTPUT_SUBDIR = "results/phase3"
DEFAULT_ALL_OUTPUT_SUBDIR = "results/all"
DEFAULT_OUTPUT_SUBDIRS = {
    "phase1": DEFAULT_PHASE1_OUTPUT_SUBDIR,
    "phase2": DEFAULT_PHASE2_OUTPUT_SUBDIR,
    "phase3": DEFAULT_PHASE3_OUTPUT_SUBDIR,
    "all": DEFAULT_ALL_OUTPUT_SUBDIR,
}

STAGE_ORDER = PHASE_ORDER
VALID_MODES = (*PHASE_ORDER, "all")
