"""Guard against the README drifting away from the measured results."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "resnet18" / "metrics.json"
PIPELINE = ROOT / "results" / "resnet18" / "pipeline_test.json"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


@pytest.mark.skipif(not METRICS.exists(), reason="no evaluation results yet")
def test_readme_headline_numbers_match_metrics():
    readme = (ROOT / "README.md").read_text()
    m = json.loads(METRICS.read_text())
    assert pct(m["tta_hflip"]["accuracy"]) in readme
    assert pct(m["single"]["accuracy"]) in readme
    assert f"{m['tta_hflip']['macro_f1']:.3f}" in readme
    for name, cls in m["tta_hflip"]["per_class"].items():
        assert f"| {name.capitalize()} | {cls['precision']:.3f} | {cls['recall']:.3f} | {cls['f1-score']:.3f} |" in readme


@pytest.mark.skipif(not PIPELINE.exists(), reason="no pipeline results yet")
def test_readme_pipeline_numbers_match():
    readme = (ROOT / "README.md").read_text()
    p = json.loads(PIPELINE.read_text())
    assert f"{p['detected']:,} / {p['images']:,}" in readme
    assert pct(p["acc_dataset_crops_on_detected"]) in readme
    assert pct(p["acc_pipeline_by_margin"]["0.9"]) in readme
