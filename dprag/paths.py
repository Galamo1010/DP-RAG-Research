"""The single owner of "where things live".

Every other module asks this one for a location instead of computing its own
relative path. That means a future move only has to be reflected here, and it
fixes a latent bug: results used to be written to a cwd-relative "results/", so
the same script wrote to different places depending on where it was launched
from.

Layout assumed (holds on Windows locally and on RunPod):

    <parent>/
    ├── dp-rag/            <- REPO_ROOT (this package lives in dp-rag/dprag/)
    │   └── results/       <- RESULTS_DIR
    └── ChatDoctor-main/   <- DATA_DIR

Both locations can be overridden by environment variable, which is how you point
a Pod at a different layout without touching code:

    CHATDOCTOR_DIR=/workspace/ChatDoctor-main
    DPRAG_RESULTS_DIR=/workspace/results
"""

from __future__ import annotations

import os
from pathlib import Path

# dp-rag/dprag/paths.py -> parents[1] is dp-rag/
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.environ.get("CHATDOCTOR_DIR") or REPO_ROOT.parent / "ChatDoctor-main"
)

RESULTS_DIR = Path(os.environ.get("DPRAG_RESULTS_DIR") or REPO_ROOT / "results")

HEALTHCAREMAGIC_PATH = DATA_DIR / "HealthCareMagic-100k.json"
ICLINIQ_PATH = DATA_DIR / "iCliniq-10k.json"


def require_data() -> Path:
    """Return DATA_DIR, failing with an actionable message if it is missing.

    Import stays side-effect free -- callers that only need config or strategies
    should not blow up because the dataset is absent. Data loaders call this.
    """
    missing = [p for p in (HEALTHCAREMAGIC_PATH, ICLINIQ_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ChatDoctor data not found.\n"
            f"  looked in : {DATA_DIR}\n"
            f"  missing   : {', '.join(p.name for p in missing)}\n"
            "Place ChatDoctor-main next to the repo, or set CHATDOCTOR_DIR."
        )
    return DATA_DIR


def results_dir() -> Path:
    """RESULTS_DIR, created on demand."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
