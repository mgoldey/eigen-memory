"""Canonical locations for committed experiment artifacts.

Results used to live at the repo root; they moved under results/ once there
were ~30 of them. Drivers should build output paths from these helpers rather
than hardcoding a directory, so a reorganization stays a one-file change.

Each helper creates its directory on demand, so a fresh clone can run any
driver without mkdir-ing first.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS = ROOT / "results"
FLIP = RESULTS / "flip"           # the label-flip experiment (act two)
SHIFT = RESULTS / "shift"         # the Rule-Shift experiment (act three)
STATIC = RESULTS / "static"       # number-game + TREC
CALIBRATION = RESULTS / "calibration"  # gate ROC sweeps + RFmu qualification
FIGURES = ROOT / "figures"


def _in(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def flip(name: str) -> Path:
    return _in(FLIP, name)


def shift(name: str) -> Path:
    return _in(SHIFT, name)


def static(name: str) -> Path:
    return _in(STATIC, name)


def calibration(name: str) -> Path:
    return _in(CALIBRATION, name)


def figure(name: str) -> Path:
    return _in(FIGURES, name)
