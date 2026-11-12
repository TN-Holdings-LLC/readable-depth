"""Tests of readable_depth.score: the rule on hand-made cases, and against a committed device record."""
import json
import os

import pytest

from readable_depth.score import readable_depth, verdict

HERE = os.path.dirname(os.path.abspath(__file__))


def test_zeros_do_not_read():
    # Returning zeros has MSE equal to the mean square: half the signal is not kept.
    assert verdict([0, 0], [1, -1])["reads"] == "DOES NOT READ"


def test_line_is_inclusive():
    # MSE exactly half the mean square reads.
    r = verdict([1.0, 0.0], [1.0, 1.0])
    assert r["mse"] == 0.5 and r["mean_square"] == 1.0 and r["reads"] == "READS"


def test_agrees_count():
    assert verdict([0.96, 0.5, -0.9], [1.0, 0.5, -1.0])["agrees"] == 2


def test_depth_per_family():
    rows = [dict(family="a", depth=d, verdict=v) for d, v in ((24, "READS"), (48, "READS"), (72, "DOES NOT READ"))]
    rows += [dict(family="b", depth=d, verdict=v) for d, v in ((8, "DOES NOT READ"), (16, "READS"))]
    out = readable_depth(rows)
    assert out["a"] == dict(readable_depth=48, first_not_reading=72, monotone=True, depths=[24, 48, 72])
    assert out["b"]["readable_depth"] == 16 and out["b"]["monotone"] is False


def test_committed_record():
    # psf-os-lab-notebook results/hub-dt-2026-11-12 (ibm_fez): the stored values are rounded to 4 decimals, so the
    # recomputed MSE matches within 1e-4 and the declared verdict is unchanged.
    data = json.load(open(os.path.join(HERE, "fixtures", "hub_dt_2026-11-12.json")))
    for tag, c in data["circuits"].items():
        r = verdict(c["z"], c["exact"])
        assert r["mse"] == pytest.approx(c["mse"], abs=1e-4), tag
        assert r["reads"] == c["reads"], tag
        assert r["agrees"] == pytest.approx(c["agrees"], abs=1), tag
