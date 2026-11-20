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
instead of extrapolating linearly forever.

**Regime gate.** With fewer than two events, or a player count that never moved,
the model takes the FLAT branch. Otherwise it takes LIVE.

**FLAT — count-consistent dilution.** The disclosed launch refresh moved the
page from 638 to 3,868 tracked players. Because earners are whole people, the
model holds the implied earner count fixed (`E = round(rate/100 × N_last)`) and
dilutes it across the growth factor `G = 3868/638`. That diluted rate and a
floor of `max(rate, 0.05)` become a two-component normal mixture on the logit
scale — weight **0.95** at **sigma 0.10** on the diluted rate, weight **0.05**
at **sigma 0.15** on the floor — and each output quantile is read off that
mixture by 60 rounds of bisection on [-16, 16].

**LIVE — saturating log-time fits.** Players are fit with Theil-Sen against `x`;
two anchored extrapolations (factors 0.85 and 1.0) are damped by a **slope
damping of 0.55** and their median becomes `N_tgt`. Each trophy's implied earner
count is fit the same way, then its raw slope is shrunk toward its
`(tier, last-rate band)` group median with weight `n_events/(n_events + 6)`; the
bands are [0,1), [1,10), [10,50), [50,100]. The trend median moves at most
**-2.2 / +1.6** logit units from the last read and is capped at 99.9% of `N_tgt`.

The median is then pulled back toward the last observed rate by a persistence
weight `lambda = min(0.97, 0.72 + 0.35 × exp(-horizon_days/2))` — **base 0.72,
gain 0.35, clamped at 0.97**. The spread combines a 0.10 floor, 0.16² × horizon
days, 0.25 × the residual scatter (clamped to [0.12, 0.6]), and a non-negative
disagreement term `lambda(1-lambda)(trend - last)²`; when the implied earner
count is below one, sigma is floored at 1.2.

**Boundaries and fallbacks.** Any trophy already at a rate ≥ 99.99 is pinned to
`[99.0, 99.9, 100.0, 100.0, 100.0]`. Every branch is wrapped: an internal
failure falls back to a persistence forecast (last rates, logit spread
`max(0.35, 0.5·√horizon_days)`), each LIVE row falls back individually, and every
result is validated (ids match, five finite monotone quantiles in [0, 100])
before it is written.

## Usage

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` beside it in
the output directory.
