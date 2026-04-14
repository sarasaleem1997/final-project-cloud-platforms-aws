import csv
import json
from pathlib import Path

import pytest

from src.diagnose import diagnose

API_DIR = Path(__file__).parent.parent
REFS_PATH = API_DIR / "reference_times.json"
CSV_PATH = API_DIR / "validation_pieces.csv"
EXPECTED_PATH = API_DIR / "validation_expected.json"

with open(REFS_PATH) as f:
    REFS = json.load(f)

MATRICES = [4974, 5052, 5090, 5091]

SEGMENT_KEYS = [
    "furnace_to_2nd_strike",
    "2nd_to_3rd_strike",
    "3rd_to_4th_strike",
    "4th_strike_to_aux_press",
    "aux_press_to_bath",
]


def _build_piece(matrix: int, overrides: dict) -> dict:
    """Build a piece dict with all partial times at reference, then apply overrides."""
    rv = REFS[str(matrix)]

    # Compute cumulative times from the 5 partial values
    partials = {k: rv[k] for k in SEGMENT_KEYS}
    partials.update(overrides)

    f = partials["furnace_to_2nd_strike"]
    t23 = partials["2nd_to_3rd_strike"]
    t34 = partials["3rd_to_4th_strike"]
    t4a = partials["4th_strike_to_aux_press"]
    tab = partials["aux_press_to_bath"]

    l2 = f
    l3 = None if (l2 is None or t23 is None) else round(l2 + t23, 1)
    l4 = None if (l3 is None or t34 is None) else round(l3 + t34, 1)
    la = None if (l4 is None or t4a is None) else round(l4 + t4a, 1)
    lb = None if (la is None or tab is None) else round(la + tab, 1)

    return {
        "piece_id": "T",
        "die_matrix": matrix,
        "lifetime_2nd_strike_s": l2,
        "lifetime_3rd_strike_s": l3,
        "lifetime_4th_strike_s": l4,
        "lifetime_auxiliary_press_s": la,
        "lifetime_bath_s": lb,
    }


# ── 24 unit tests: 4 matrices × 6 scenarios ──────────────────────────────────

@pytest.mark.parametrize("matrix", MATRICES)
def test_all_ok(matrix):
    piece = _build_piece(matrix, {})
    result = diagnose(piece, REFS)
    assert result["delay"] is False
    assert result["probable_causes"] == []
    for s in result["segments"]:
        assert s["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_furnace_to_2nd_penalized(matrix):
    rv = REFS[str(matrix)]
    piece = _build_piece(matrix, {"furnace_to_2nd_strike": rv["furnace_to_2nd_strike"] + 2.0})
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["furnace_to_2nd_strike"]["penalized"] is True
    for seg in SEGMENT_KEYS[1:]:
        assert segs[seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_2nd_to_3rd_penalized(matrix):
    rv = REFS[str(matrix)]
    piece = _build_piece(matrix, {"2nd_to_3rd_strike": rv["2nd_to_3rd_strike"] + 2.0})
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["2nd_to_3rd_strike"]["penalized"] is True
    for seg in [k for k in SEGMENT_KEYS if k != "2nd_to_3rd_strike"]:
        assert segs[seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_3rd_to_4th_penalized(matrix):
    rv = REFS[str(matrix)]
    piece = _build_piece(matrix, {"3rd_to_4th_strike": rv["3rd_to_4th_strike"] + 2.0})
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["3rd_to_4th_strike"]["penalized"] is True
    for seg in [k for k in SEGMENT_KEYS if k != "3rd_to_4th_strike"]:
        assert segs[seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_4th_to_aux_penalized(matrix):
    rv = REFS[str(matrix)]
    piece = _build_piece(matrix, {"4th_strike_to_aux_press": rv["4th_strike_to_aux_press"] + 2.0})
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["4th_strike_to_aux_press"]["penalized"] is True
    for seg in [k for k in SEGMENT_KEYS if k != "4th_strike_to_aux_press"]:
        assert segs[seg]["penalized"] is False


@pytest.mark.parametrize("matrix", MATRICES)
def test_aux_to_bath_penalized(matrix):
    rv = REFS[str(matrix)]
    piece = _build_piece(matrix, {"aux_press_to_bath": rv["aux_press_to_bath"] + 2.0})
    result = diagnose(piece, REFS)
    assert result["delay"] is True
    segs = {s["segment"]: s for s in result["segments"]}
    assert segs["aux_press_to_bath"]["penalized"] is True
    for seg in [k for k in SEGMENT_KEYS if k != "aux_press_to_bath"]:
        assert segs[seg]["penalized"] is False


# ── Golden test: all 10 validation pieces ────────────────────────────────────

def _load_validation_pieces():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            piece = {
                "piece_id": row["piece_id"],
                "die_matrix": int(row["die_matrix"]),
            }
            for col in [
                "lifetime_2nd_strike_s",
                "lifetime_3rd_strike_s",
                "lifetime_4th_strike_s",
                "lifetime_auxiliary_press_s",
                "lifetime_bath_s",
            ]:
                piece[col] = float(row[col]) if row[col] != "" else None
            rows.append(piece)
    return rows


def _round_result(result: dict) -> dict:
    """Round all float fields to 1 decimal for comparison."""
    out = dict(result)
    out["segments"] = []
    for s in result["segments"]:
        seg = dict(s)
        for field in ("actual_s", "reference_s", "deviation_s"):
            if seg[field] is not None:
                seg[field] = round(seg[field], 1)
        out["segments"].append(seg)
    return out


validation_pieces = _load_validation_pieces()
with open(EXPECTED_PATH) as f:
    expected_outputs = json.load(f)

ids_and_pieces = [(p["piece_id"], p, e) for p, e in zip(validation_pieces, expected_outputs)]


@pytest.mark.parametrize("piece_id,piece,expected", ids_and_pieces, ids=[x[0] for x in ids_and_pieces])
def test_golden(piece_id, piece, expected):
    result = diagnose(piece, REFS)
    assert _round_result(result) == _round_result(expected)
