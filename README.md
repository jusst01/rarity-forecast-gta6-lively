# Gentle-Shrinkage Rarity Forecaster — GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/jusst01/rarity-forecast-gta6-lively/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/jusst01/rarity-forecast-gta6-lively/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

[Competition Page](https://make.fun/c/gta6-trophy-rarity-forecast-01)

## Model Overview

This entry forecasts each GTA VI trophy's Exophase earn rate at a target time one
week out, as five quantiles (5%, 25%, 50%, 75%, 95%). It is pure Python standard
library, deterministic, and never raises.

**Eventizing.** The refresh history is collapsed into distinct states, oldest
first: a new event begins whenever the tracked-player count or the rate tuple
changes. Every event is placed on a saturating time axis, `x = ln(1 + hours
since the first read)`, which lets a rate that is still climbing flatten out
instead of extrapolating linearly forever. Quantile z values are
`[-1.644854, -0.674490, 0, 0.674490, 1.644854]`.

**Regime gate.** With fewer than two events, or a player count that never moved,
the model takes the FLAT branch. Otherwise it takes LIVE.

**FLAT — count-consistent dilution.** The disclosed launch refresh moved the
page from 638 to 3,868 tracked players, so the growth factor is `G = 3868/638`.
Because earners are whole people, the model holds the implied earner count fixed
(`E = round(rate/100 × N_last)`) and dilutes it to `100·E/(N_last·G)`; if `E` is
zero the diluted rate is the **0.02** floor. That diluted rate and
`max(rate, 0.05)` become a two-component normal mixture on the logit scale —
weight **0.95** at **sigma 0.10** on the diluted rate, weight **0.05** at
**sigma 0.15** on the floor — and each output quantile is read off that mixture
by **60 rounds** of bisection on **[-16, 16]** using `math.erf`'s normal CDF,
then mapped back through the inverse logit and sorted.

**LIVE — saturating log-time fits.** Players are fit with Theil-Sen on
`ln(max(N, 10))` against `x`; two anchored extrapolations (factors **0.85** and
**1.0**) are damped by a **slope damping of 0.55** and their median becomes
`N_tgt`. Each trophy's implied earner count is fit the same way on
`ln(max(0.5, round(rate/100 × N)))`, then its raw slope is shrunk toward its
`(tier, last-rate band)` group median with weight `w = n_events/(n_events + 6)`;
the bands are **[0,1), [1,10), [10,50), [50,100]**, and a missing group uses the
global median slope. The earner trend applies the same **0.55** damping, is
capped at **99.9% of `N_tgt`**, and the resulting trend median's logit may move
from the last read's logit by at most **[-2.2, +1.6]**.

The median is then pulled back toward the last observed rate by a persistence
weight

```
lambda = min(0.97, 0.72 + 0.35 × exp(-horizon_days/2)),   horizon_days = horizon_minutes/1440
```

— **base 0.72, gain 0.35, clamped at 0.97** (the clamp is load-bearing: a lambda
above 1 would make the disagreement term negative). The spread is

```
sigma = sqrt(0.10² + 0.16²·horizon_days + 0.25·rsd² + lambda·(1-lambda)·(trend_logit - last_logit)²)
```

where `rsd` is the root-mean-square residual of a Theil-Sen fit of the event
logit-rates, **clamped to [0.12, 0.6]**, and the disagreement term is guarded
non-negative. When the last implied earner count is **below 1**, sigma is floored
at **1.2**.

**Boundaries and fallbacks.** Any trophy already at a rate ≥ **99.99** is pinned
to `[99.0, 99.9, 100.0, 100.0, 100.0]` in both branches. Every branch is wrapped:
an internal failure falls back to a persistence forecast (last rates, logit
spread `max(0.35, 0.5·√horizon_days)`), each LIVE row falls back individually,
and every result is validated (ids match the case, five finite monotone
quantiles in [0, 100]) before it is written. A file that fails to parse as JSON
still gets a persistence output when its `trophies` array can be salvaged, so a
truncated or malformed case never goes unanswered. Output is rounded to 4
decimals (stepping down to 2, 1 or 0 only if that would exceed the 65,536-byte
contract) and is byte-identical across runs.

## Usage

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` beside it in
the output directory.
