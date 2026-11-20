"""Gentle-Shrinkage Rarity Forecaster - gta6-trophy-rarity-forecast-01.

Forecasts each GTA VI trophy's earn rate on Exophase at the case's target: one
week (or whatever horizon the case sets) after its last read. Two regimes:

  * FLAT - fewer than two distinct refresh states, or the tracked-player count
           never moved. The disclosed refresh diluted the 638-player page across
           3,868 tracked players, so we hold the integer earner count fixed and
           dilute its rate, then spread it as a two-component normal mixture on
           the logit scale.

  * LIVE - the tracked-player count moved. We fit saturating log-time trends
           (Theil-Sen against x = ln(1 + hours since the first read)) for players
           and earners, damp the slopes, cap the median's logit move, and shrink
           the median back toward the last observed rate by a horizon-dependent
           persistence weight.

Python standard library only, deterministic, never raises.
"""

import datetime
import json
import math
import os
import sys
from pathlib import Path

# The grader runs `python3 -P -s`: -P keeps this file's directory OFF sys.path,
# so put it back first (a module shipped beside main.py must be importable).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# This entry's calibration constants (deliberately gentler than sibling entries).
SLOPE_DAMPING = 0.55          # log-time slope damping for players and earners
PERSISTENCE_BASE = 0.72       # lambda base
PERSISTENCE_GAIN = 0.35       # lambda gain on exp(-horizon_days / 2)
PERSISTENCE_CAP = 0.97        # load-bearing clamp; lambda > 1 would flip the disagreement term
MOVE_DOWN = -2.2              # most the trend median's logit may fall below last
MOVE_UP = 1.6                 # most it may rise above last
GROWTH = 3868.0 / 638.0       # disclosed launch-refresh tracked-player growth factor

PIN_AT = 99.99
PINNED = [99.0, 99.9, 100.0, 100.0, 100.0]

BANDS = [(0.0, 1.0), (1.0, 10.0), (10.0, 50.0), (50.0, 100.0)]


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    """logit of a percentage: clamp p to [0.01, 99.99], then ln((p/100)/(1-p/100))."""
    p = clamp(float(p), 0.01, 99.99)
    q = p / 100.0
    return math.log(q / (1.0 - q))


def inv_logit(l):
    """100/(1+exp(-l)), guarded against overflow for large |l|."""
    l = float(l)
    if l > 700.0:
        return 100.0
    if l < -700.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def theil_sen(xs, ys):
    """(slope, intercept) by Theil-Sen; pairs need distinct x, degenerate gives slope 0."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    slopes = []
    for i in range(len(pts)):
        xi, yi = pts[i]
        for j in range(i + 1, len(pts)):
            xj, yj = pts[j]
            if xj != xi:
                slopes.append((yj - yi) / (xj - xi))
    slope = median(slopes) if slopes else 0.0
    intercept = median([y - slope * x for x, y in pts]) if pts else 0.0
    return slope, intercept


def rhu(x, places=4):
    """Round half up to `places` decimals, deterministically (JS-Math.round-like)."""
    scale = 10.0 ** places
    scaled = float(x) * scale
    if not math.isfinite(scaled):
        return 0.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / scale


def count_round(x):
    """Non-negative half-up integer rounding of an implied earner count."""
    if not math.isfinite(x) or x <= 0.0:
        return 0
    return int(math.floor(x + 0.5))


def safe_rate(value):
    """A printed rate as a finite float; nulls, junk and non-finite values read as 0.0."""
    if value is None:
        return 0.0
    try:
        r = float(value)
    except (TypeError, ValueError):
        return 0.0
    return r if math.isfinite(r) else 0.0


def parse_ts(value):
    """ISO-8601 timestamp -> epoch seconds, or None if it cannot be parsed."""
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


# --------------------------------------------------------------------------- #
# Eventizing the refresh history
# --------------------------------------------------------------------------- #
def eventize(history):
    """Collapse reads into distinct states, oldest first.

    A new event starts whenever (players_tracked, rates tuple) differs from the
    previous read. Nulls stay in the comparison key (as a distinct marker) but
    read as 0.0 in the numeric values. Each event carries hours since the first
    read's timestamp.
    """
    events = []
    sentinel = object()
    prev_key = sentinel
    first_stamp = None
    for idx, read in enumerate(history):
        if not isinstance(read, dict):
            read = {}
        try:
            players = float(read.get("players_tracked"))
        except (TypeError, ValueError):
            players = 0.0
        raw = read.get("rates")
        if not isinstance(raw, list):
            raw = []
        key = (players, tuple(raw))
        if prev_key is not sentinel and key == prev_key:
            continue
        prev_key = key

        stamp = parse_ts(read.get("at"))
        if stamp is None:
            stamp = float(idx) * 3600.0  # unparseable clock: one hour per read index
        if first_stamp is None:
            first_stamp = stamp
        hours = max(0.0, (stamp - first_stamp) / 3600.0)

        values = [safe_rate(r) for r in raw]
        events.append({"hours": hours, "players": players, "rates": values})
    return events


def event_x(events):
    """x = ln(1 + hours since first read) - the time axis for every fit."""
    return [math.log1p(ev["hours"]) for ev in events]


# --------------------------------------------------------------------------- #
# Persistence fallback (used by every guard)
# --------------------------------------------------------------------------- #
def last_read_rates(case, n_trophies):
    history = case.get("history")
    if not isinstance(history, list) or not history:
        return [0.0] * n_trophies
    last = history[-1]
    raw = last.get("rates") if isinstance(last, dict) else None
    if not isinstance(raw, list):
        return [0.0] * n_trophies
    out = []
    for i in range(n_trophies):
        out.append(safe_rate(raw[i]) if i < len(raw) else 0.0)
    return out


def persist_rows(case):
    """Persistence forecast: last rates with a logit spread of max(0.35, 0.5*sqrt(d))."""
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        return []
    try:
        horizon_days = max(0.0, float(case.get("horizon_minutes", 0)) / 1440.0)
    except (TypeError, ValueError, OverflowError):
        horizon_days = 0.0
    spread = max(0.35, 0.5 * math.sqrt(horizon_days))
    rates = last_read_rates(case, len(trophies))
    rows = []
    for i, trophy in enumerate(trophies):
        rate = rates[i]
        if rate >= PIN_AT:
            q = list(PINNED)
        else:
            lg = logit_pct(rate)
            q = sorted(clamp(inv_logit(lg + z * spread), 0.0, 100.0) for z in Z)
        rows.append({"id": trophy.get("id") if isinstance(trophy, dict) else None, "q": q})
    return rows


# --------------------------------------------------------------------------- #
# FLAT branch - count-consistent dilution
# --------------------------------------------------------------------------- #
def mixture_quantile(level, mu1, s1, w1, mu2, s2):
    """Quantile of w1*N(mu1,s1) + (1-w1)*N(mu2,s2) by deterministic bisection."""
    w2 = 1.0 - w1

    def cdf(t):
        return w1 * norm_cdf((t - mu1) / s1) + w2 * norm_cdf((t - mu2) / s2)

    lo, hi = -16.0, 16.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def flat_row(rate, players):
    """Dilute a flat history across the known 638 -> 3,868 player refresh.

    N_last is the last event's player count exactly as read. When it is
    non-positive the implied earner count is zero, so the diluted rate takes
    the 0.02 floor (the spec has no substitute player count).
    """
    n_last = float(players)
    earners = count_round(float(rate) / 100.0 * n_last) if n_last > 0.0 else 0
    diluted = 100.0 * earners / (n_last * GROWTH) if earners > 0 else 0.02
    mu1 = logit_pct(diluted)
    mu2 = logit_pct(max(float(rate), 0.05))
    out = [clamp(inv_logit(mixture_quantile(level, mu1, 0.10, 0.95, mu2, 0.15)), 0.0, 100.0)
           for level in LEVELS]
    return sorted(out)


# --------------------------------------------------------------------------- #
# LIVE branch - saturating log-time fits
# --------------------------------------------------------------------------- #
def band_of(rate):
    """Last-rate band of the shrinkage group; rates clamp into [0, 100] first."""
    r = clamp(float(rate), 0.0, 100.0)
    for lo, hi in BANDS:
        if lo <= r < hi:
            return (lo, hi)
    return BANDS[-1]


def live_context(events, trophies, horizon_minutes):
    """Shared LIVE quantities: time axis, lambda, N_tgt, per-trophy slopes, group medians."""
    xs = event_x(events)
    x_last = xs[-1]
    x_tgt = math.log1p(events[-1]["hours"] + float(horizon_minutes) / 60.0)

    horizon_days = max(0.0, float(horizon_minutes) / 1440.0)
    lam = min(PERSISTENCE_CAP,
              PERSISTENCE_BASE + PERSISTENCE_GAIN * math.exp(-horizon_days / 2.0))

    # Players: Theil-Sen on ln(max(N, 10)); median of two damped anchored extrapolations.
    ys_players = [math.log(max(ev["players"], 10.0)) for ev in events]
    b_n, _ = theil_sen(xs, ys_players)
    n_last = max(events[-1]["players"], 1e-9)
    anchors = [n_last * math.exp(f * SLOPE_DAMPING * b_n * (x_tgt - x_last))
               for f in (0.85, 1.0)]
    n_tgt = median(anchors)
    if not math.isfinite(n_tgt) or n_tgt <= 0.0:
        n_tgt = max(n_last, 1e-9)

    n_events = len(events)
    w = n_events / (n_events + 6.0)  # group-shrinkage weight on the raw slope

    raw_slopes = {}
    rsd_by_trophy = {}
    for i, trophy in enumerate(trophies):
        rates = [ev["rates"][i] if i < len(ev["rates"]) else 0.0 for ev in events]
        ys_earn = [math.log(max(0.5, float(count_round(r / 100.0 * ev["players"]))))
                   for r, ev in zip(rates, events)]
        raw_slopes[i], _ = theil_sen(xs, ys_earn)

        ys_logit = [logit_pct(max(r, 0.01)) for r in rates]
        s, icpt = theil_sen(xs, ys_logit)
        resid = [y - (s * x + icpt) for x, y in zip(xs, ys_logit)]
        rsd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.12
        rsd_by_trophy[i] = clamp(rsd, 0.12, 0.6)

    # Group by (tier, last-rate band); a missing group falls back to the global median,
    # taken over the same non-pinned trophies that populate the groups.
    groups = {}
    for i, trophy in enumerate(trophies):
        rates = [ev["rates"][i] if i < len(ev["rates"]) else 0.0 for ev in events]
        if rates[-1] >= PIN_AT:
            continue
        key = (trophy.get("tier") if isinstance(trophy, dict) else None, band_of(rates[-1]))
        groups.setdefault(key, []).append(raw_slopes[i])
    pooled = [slope for slopes in groups.values() for slope in slopes]
    global_median = median(pooled) if pooled else 0.0

    return {
        "x_last": x_last, "x_tgt": x_tgt, "horizon_days": horizon_days, "lam": lam,
        "n_last": n_last, "n_tgt": n_tgt, "w": w,
        "raw_slopes": raw_slopes, "rsd": rsd_by_trophy,
        "groups": {k: median(v) for k, v in groups.items()},
        "global": global_median,
    }


def live_row(i, trophy, events, ctx):
    """One LIVE trophy row; raises on internal failure so the caller can fall back."""
    rates = [ev["rates"][i] if i < len(ev["rates"]) else 0.0 for ev in events]
    last_rate = rates[-1]
    if last_rate >= PIN_AT:
        return list(PINNED)

    tier = trophy.get("tier") if isinstance(trophy, dict) else None
    group_median = ctx["groups"].get((tier, band_of(last_rate)), ctx["global"])
    b_shrunk = ctx["w"] * ctx["raw_slopes"][i] + (1.0 - ctx["w"]) * group_median

    x_last, x_tgt = ctx["x_last"], ctx["x_tgt"]
    n_last, n_tgt = ctx["n_last"], ctx["n_tgt"]
    lam, horizon_days = ctx["lam"], ctx["horizon_days"]

    # Trend median on the earner-count scale, damped, capped at 99.9% of N_tgt.
    e_last = count_round(last_rate / 100.0 * n_last)
    ln_e_tgt = math.log(max(e_last, 0.5)) + SLOPE_DAMPING * b_shrunk * (x_tgt - x_last)
    try:
        e_tgt = min(math.exp(ln_e_tgt), 0.999 * n_tgt)
    except OverflowError:
        e_tgt = 0.999 * n_tgt
    if not math.isfinite(e_tgt) or e_tgt < 0.0:
        e_tgt = 0.0
    trend_rate = 100.0 * e_tgt / n_tgt
    trend_rate = clamp(trend_rate, 1e-9, 100.0)

    # Movement cap on the logit scale, then persistence shrinkage toward last.
    last_logit = logit_pct(max(last_rate, 0.05))
    trend_logit = clamp(logit_pct(trend_rate), last_logit + MOVE_DOWN, last_logit + MOVE_UP)
    m = (1.0 - lam) * trend_logit + lam * last_logit

    # Spread; the disagreement term is guarded non-negative.
    rsd = ctx["rsd"].get(i, 0.12)
    disagreement = max(0.0, lam * (1.0 - lam) * (trend_logit - last_logit) ** 2)
    sigma = math.sqrt(0.10 ** 2 + 0.16 ** 2 * horizon_days + 0.25 * rsd ** 2 + disagreement)
    if e_last < 1:
        sigma = max(sigma, 1.2)

    return sorted(clamp(inv_logit(m + z * sigma), 0.0, 100.0) for z in Z)


def live_rows(events, trophies, horizon_minutes, case):
    ctx = live_context(events, trophies, horizon_minutes)
    fallback = persist_rows(case)
    rows = []
    for i, trophy in enumerate(trophies):
        try:
            q = live_row(i, trophy, events, ctx)
        except Exception:
            q = fallback[i]["q"]
        rows.append({"id": trophy.get("id") if isinstance(trophy, dict) else None, "q": q})
    return rows


def use_flat_branch(events):
    """Regime gate: FLAT with fewer than two events or a single player count."""
    if len(events) < 2:
        return True
    first = events[0]["players"]
    return all(ev["players"] == first for ev in events[1:])


# --------------------------------------------------------------------------- #
# Validation and finalisation
# --------------------------------------------------------------------------- #
def validate(rows, case):
    """Ids match the case, five finite monotone quantiles each, all inside [0, 100]."""
    trophies = case.get("trophies")
    if not isinstance(trophies, list) or not isinstance(rows, list):
        return False
    if len(rows) != len(trophies):
        return False
    for row, trophy in zip(rows, trophies):
        if not isinstance(row, dict):
            return False
        want = trophy.get("id") if isinstance(trophy, dict) else None
        if row.get("id") != want:
            return False
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            return False
        prev = -1.0
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return False
            if v < 0.0 or v > 100.0 or v < prev:
                return False
            prev = v
    return True


def finalize(rows):
    """Round to 4 decimals, clamp, sort; emit the contract's dict.

    Rounding is monotone, so sorted rows stay non-decreasing. If the encoded
    result would exceed the 65,536-byte contract, precision steps down to 2, 1
    and 0 decimals rather than dropping any trophy row.
    """
    for places in (4, 2, 1, 0):
        clean = []
        for row in rows:
            q = sorted(rhu(clamp(float(v), 0.0, 100.0), places) for v in row["q"])
            clean.append({"id": row["id"], "q": q})
        out = {"trophies": clean}
        if len(json.dumps(out, separators=(",", ":")).encode("utf-8")) <= 65536:
            return out
    return out


def salvage_trophies(text):
    """Best-effort recovery of the `trophies` array from a malformed case file.

    A truncated write usually leaves the trophies array intact while the
    history is cut off, so scan for the array's brackets (string-aware) and
    parse just that slice. Returns a list, or None when nothing usable exists.
    """
    try:
        key = text.find('"trophies"')
        if key < 0:
            return None
        start = text.find("[", key)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for pos in range(start, len(text)):
            ch = text[pos]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start:pos + 1])
                    return parsed if isinstance(parsed, list) else None
        return None
    except Exception:
        return None


def salvage_horizon(text):
    """Best-effort horizon_minutes from a malformed case file; 0.0 when absent."""
    try:
        marker = text.find('"horizon_minutes"')
        if marker < 0:
            return 0.0
        colon = text.find(":", marker)
        if colon < 0:
            return 0.0
        token = ""
        for ch in text[colon + 1:colon + 40]:
            if ch in "-+.0123456789eE":
                token += ch
            elif token:
                break
            elif not ch.isspace():
                break
        value = float(token)
        return value if math.isfinite(value) and value > 0.0 else 0.0
    except Exception:
        return 0.0


def fallback_output(case):
    """Persistence output for a case whose forecast could not be computed."""
    try:
        forecast = finalize(persist_rows(case))
        if isinstance(forecast.get("trophies"), list) and forecast["trophies"]:
            return forecast
    except Exception:
        pass
    return {"trophies": []}


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def forecast_case(case, name):
    """Forecast one case; never raises, always returns a valid row set."""
    try:
        trophies = case.get("trophies")
        if not isinstance(trophies, list):
            raise ValueError("no trophies")

        history = case.get("history")
        if not isinstance(history, list):
            history = []
        try:
            horizon = float(case.get("horizon_minutes", 0))
        except (TypeError, ValueError, OverflowError):
            horizon = 0.0
        if not math.isfinite(horizon):
            horizon = 0.0
        horizon = max(0.0, horizon)

        events = eventize(history)

        if use_flat_branch(events):
            # FLAT branch: count-consistent dilution.
            last = events[-1] if events else {"players": 0.0, "rates": []}
            n_last = last["players"]
            rows = []
            for i, trophy in enumerate(trophies):
                rate = last["rates"][i] if i < len(last["rates"]) else 0.0
                tid = trophy.get("id") if isinstance(trophy, dict) else None
                if rate >= PIN_AT:
                    rows.append({"id": tid, "q": list(PINNED)})
                    continue
                try:
                    q = flat_row(rate, n_last)
                except Exception:
                    q = persist_rows(case)[i]["q"]
                rows.append({"id": tid, "q": q})
        else:
            # LIVE branch: saturating log-time fits.
            try:
                rows = live_rows(events, trophies, horizon, case)
            except Exception:
                rows = persist_rows(case)

        if not validate(rows, case):
            rows = persist_rows(case)
        if not validate(rows, case):
            rows = [{"id": (t.get("id") if isinstance(t, dict) else None), "q": list(PINNED)}
                    for t in trophies]
        return finalize(rows)
    except Exception:
        try:
            rows = persist_rows(case)
            if not validate(rows, case):
                rows = [{"id": (t.get("id") if isinstance(t, dict) else None), "q": list(PINNED)}
                        for t in (case.get("trophies") if isinstance(case.get("trophies"), list) else [])]
            return finalize(rows)
        except Exception:
            return {"trophies": []}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file in name order, so two runs write identical bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        case = None
        try:
            case = json.loads(text)
        except Exception:
            case = None

        if isinstance(case, dict):
            try:
                forecast = forecast_case(case, path.stem)
            except Exception:
                # Malformed case: still emit a persistence output when trophies are readable.
                forecast = fallback_output(case)
        else:
            # Unparseable file: salvage the trophy list so the case still gets rows.
            trophies = salvage_trophies(text)
            if trophies is None:
                continue
            case = {"trophies": trophies, "history": [], "horizon_minutes": salvage_horizon(text)}
            forecast = fallback_output(case)

        try:
            (OUTPUT_DIR / (path.stem + ".out")).write_text(
                json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
            )
        except Exception:
            continue


if __name__ == "__main__":
    main()
