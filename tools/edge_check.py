#!/usr/bin/env python3
"""Self-check for the Gentle-Shrinkage Rarity Forecaster.

Loads main.py beside it, runs forecast_case over the kit practice cases plus the
mutated edge cases the never-crash contract cares about, and asserts the output
contract on every row set: ids match the case in order, five finite
non-decreasing quantiles inside [0, 100], and the encoded result stays under
65,536 bytes. Determinism is checked by calling every case twice.

Usage:
    python3 tools/edge_check.py [--kit-dir PATH]

Exit status is 0 when every check passes, 1 otherwise.
"""

import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_PY = HERE.parent / "main.py"
DEFAULT_KIT = HERE.parent.parent / "kit" / "input"

MAX_BYTES = 65536


def load_main():
    """Import main.py under an isolated name so tools/ never shadows anything."""
    spec = importlib.util.spec_from_file_location("lively_main", MAIN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_rows(case, result, label):
    """Assert the full output contract for one result; return a problem list."""
    problems = []
    if not isinstance(result, dict):
        return ["%s: result is not a dict" % label]
    rows = result.get("trophies")
    if not isinstance(rows, list):
        return ["%s: no trophies list" % label]
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        return ["%s: case has no trophy list to compare against" % label]
    if len(rows) != len(trophies):
        problems.append("%s: %d rows for %d trophies" % (label, len(rows), len(trophies)))
    seen = set()
    for pos, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "q"}:
            problems.append("%s: row %d is not {id, q}" % (label, pos))
            continue
        if pos < len(trophies):
            want = trophies[pos].get("id")
            if row["id"] != want:
                problems.append("%s: row %d id %r != %r" % (label, pos, row["id"], want))
        if row["id"] in seen:
            problems.append("%s: duplicate id %r" % (label, row["id"]))
        seen.add(row["id"])
        q = row["q"]
        if not isinstance(q, list) or len(q) != 5:
            problems.append("%s: row %d q is not five values" % (label, pos))
            continue
        prev = -1.0
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append("%s: row %d has a non-finite quantile" % (label, pos))
                break
            if v < 0.0 or v > 100.0:
                problems.append("%s: row %d quantile %r outside [0, 100]" % (label, pos, v))
                break
            if v < prev:
                problems.append("%s: row %d quantiles not monotone" % (label, pos))
                break
            prev = v
    size = len(json.dumps(result, separators=(",", ":")).encode("utf-8"))
    if size > MAX_BYTES:
        problems.append("%s: %d bytes exceeds %d" % (label, size, MAX_BYTES))
    return problems


def mutated_cases(base):
    """The hostile variants of a practice case that must never crash the model."""
    def variant(fn):
        case = copy.deepcopy(base)
        fn(case)
        return case

    def empty_history(c):
        c["history"] = []

    def single_read(c):
        c["history"] = c["history"][:1] if c["history"] else []

    def null_rates(c):
        for read in c["history"]:
            n = len(read.get("rates") or [])
            read["rates"] = [None] * n

    def zero_rates(c):
        for read in c["history"]:
            n = len(read.get("rates") or [])
            read["rates"] = [0.0] * n

    def no_history_key(c):
        c.pop("history", None)

    def junk_everything(c):
        c["history"] = [{"players_tracked": None, "rates": None, "at": "not-a-time"},
                        {"players_tracked": "x", "rates": [None, "y", 1e309], "at": None}]
        c["horizon_minutes"] = "soon"

    def zero_players(c):
        for read in c["history"]:
            read["players_tracked"] = 0

    def moving_players(c):
        for k, read in enumerate(c["history"]):
            read["players_tracked"] = 638 + 300 * k

    return {
        "empty history": variant(empty_history),
        "single read": variant(single_read),
        "all-null rates": variant(null_rates),
        "all-zero rates": variant(zero_rates),
        "missing history key": variant(no_history_key),
        "junk fields": variant(junk_everything),
        "zero players": variant(zero_players),
        "moving players (LIVE)": variant(moving_players),
    }


def malformed_text_cases(base_text, base):
    """Cases whose file is unparseable but whose trophy list should be salvaged.

    Each entry is (text, expect_output). An empty file or a bare JSON scalar
    carries no trophies at all, so no output file is the correct outcome.
    """
    full = json.dumps({"trophies": base["trophies"], "history": [], "horizon_minutes": 90,
                       "levels": [0.05, 0.25, 0.5, 0.75, 0.95]})
    trunc_at = full.find('"history"')
    truncated = full[:trunc_at + len('"histor')] if trunc_at > 0 else full[: len(full) // 2]
    return {
        "truncated json": (truncated, True),
        "garbage prefix": ("{oops" + base_text, True),
        "empty file": ("", False),
        "json scalar": ("42", False),
    }


def main():
    kit = Path(sys.argv[sys.argv.index("--kit-dir") + 1]) if "--kit-dir" in sys.argv else DEFAULT_KIT
    if not kit.is_dir():
        print("SKIP: kit input dir not found at %s" % kit)
        return 1
    model = load_main()
    failures = []
    checked = 0

    print("== practice cases ==")
    cases = {}
    for path in sorted(kit.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("  %-10s UNREADABLE (%s)" % (path.name, exc))
            failures.append("%s unreadable" % path.name)
            continue
        cases[path.stem] = case
        first = model.forecast_case(case, path.stem)
        second = model.forecast_case(copy.deepcopy(case), path.stem)
        problems = check_rows(case, first, path.stem)
        if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
            problems.append("%s: not deterministic across calls" % path.stem)
        checked += 1
        status = "ok" if not problems else "FAIL"
        print("  %-10s %-4s rows=%d bytes=%d" % (
            path.name, status, len(first.get("trophies", [])),
            len(json.dumps(first, separators=(",", ":")).encode("utf-8"))))
        failures.extend(problems)

    base = cases.get("p00") or (next(iter(cases.values())) if cases else None)
    if base is None:
        print("FAIL: no practice case available to mutate")
        return 1

    print("== mutated edge cases (in-process) ==")
    for label, case in mutated_cases(base).items():
        try:
            result = model.forecast_case(case, label)
            problems = check_rows(case, result, label)
        except Exception as exc:
            problems = ["%s: raised %s: %s" % (label, type(exc).__name__, exc)]
            result = None
        checked += 1
        status = "ok" if not problems else "FAIL"
        nrows = len(result.get("trophies", [])) if isinstance(result, dict) else 0
        print("  %-24s %-4s rows=%d" % (label, status, nrows))
        failures.extend(problems)

    print("== malformed files (via main()) ==")
    base_text = json.dumps(base)
    with tempfile.TemporaryDirectory() as tmp:
        in_dir = Path(tmp) / "in"
        out_dir = Path(tmp) / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        expected = {}
        for label, (text, expect_output) in malformed_text_cases(base_text, base).items():
            name = label.replace(" ", "_")
            (in_dir / (name + ".in")).write_text(text, encoding="utf-8")
            expected[name] = (label, expect_output)
        old_in, old_out = os.environ.get("MAKEFUN_INPUT_DIR"), os.environ.get("MAKEFUN_OUTPUT_DIR")
        os.environ["MAKEFUN_INPUT_DIR"] = str(in_dir)
        os.environ["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
        try:
            model.INPUT_DIR = in_dir
            model.OUTPUT_DIR = out_dir
            model.main()
        finally:
            if old_in is None:
                os.environ.pop("MAKEFUN_INPUT_DIR", None)
            else:
                os.environ["MAKEFUN_INPUT_DIR"] = old_in
            if old_out is None:
                os.environ.pop("MAKEFUN_OUTPUT_DIR", None)
            else:
                os.environ["MAKEFUN_OUTPUT_DIR"] = old_out
        for name, (label, expect_output) in expected.items():
            out = out_dir / (name + ".out")
            checked += 1
            if not expect_output:
                if out.is_file():
                    print("  %-24s FAIL unexpected output written" % label)
                    failures.append("%s: unexpected output file" % label)
                else:
                    print("  %-24s ok   (no trophies to salvage, no output)" % label)
                continue
            if not out.is_file():
                print("  %-24s FAIL no output written" % label)
                failures.append("%s: no output file" % label)
                continue
            try:
                result = json.loads(out.read_text(encoding="utf-8"))
            except Exception as exc:
                print("  %-24s FAIL unreadable output (%s)" % (label, exc))
                failures.append("%s: unreadable output" % label)
                continue
            salvaged = {"trophies": base["trophies"]}
            problems = check_rows(salvaged, result, label)
            status = "ok" if not problems else "FAIL"
            print("  %-24s %-4s rows=%d" % (label, status, len(result.get("trophies", []))))
            failures.extend(problems)

    print()
    if failures:
        print("EDGE CHECK FAILED (%d checks, %d problems)" % (checked, len(failures)))
        for problem in failures[:20]:
            print("  -", problem)
        return 1
    print("EDGE CHECK PASSED: %d cases, all rows valid, monotone, in range, deterministic" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
