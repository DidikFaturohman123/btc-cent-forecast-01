# btc-cent-forecast-01 entry: v3 "volscale" (fresh re-author, 2026-11-15).
#
# Model (from the prose design in REPORT-V2.md section 5):
#   forecast = last * exp(-0.20 * dev240 * (v1440 / v240))
# where, for a case holding the last 4320 one-minute closes ending at the
# checkpoint (last = close at the checkpoint):
#   dev240 = log(last / mean240), mean240 = mean of the last 240 closes
#            (inclusive of last) -- a single-feature mean-reversion term:
#            price above its 240-minute mean is pulled down, below it up.
#   v1440  = sample standard deviation of log returns over the last 1440
#            minutes (1440 returns, prices[-1441:]).
#   v240   = sample standard deviation of log returns over the last 240
#            minutes (240 returns, prices[-241:]).
#   The ratio v1440 / v240 scales the reversion: when the long window was
#   choppier than the recent window the pull is stronger, and vice versa.
# Parameters: p = +1.0 (full-strength ratio, not damped), k = 0.20.
#
# Uses only in-case data, Python 3.13 standard library only, deterministic
# (no randomness, no time, no I/O beyond the case files), runs in well under
# a second per case (limit 8 s).
#
# Validated fresh on the kit practice cases p00-p05 (2026-11-15, builder
# session): practice MAE 43,892.69 cents vs persistence 44,521.58 (-1.41%),
# matching the REPORT-V2.md target 43,892.49 within 0.20 cents; two runs
# byte-identical; ~0.013 s per case.
#
# I/O protocol (competition.yaml / starter/main.ts): read every <id>.in in
# the input directory ($MAKEFUN_INPUT_DIR, else /input), each a JSON case
# with one market ("btc-usdc") holding `prices`; write <id>.out to the
# output directory ($MAKEFUN_OUTPUT_DIR, else /output) as
# {"forecasts": [{"case": "<id>", "price": <number>}]}, the price kept on
# the grid's 5 decimals and inside (0, 10,000,000), exactly like the
# starter's price5.

import json
import math
import os
import statistics
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# v3 volscale parameters.
K = 0.20
P = 1.0  # exponent on the volatility ratio (full strength)


def price5(x):
    """x kept on the price grid's 5 decimals and inside (0, 10,000,000)."""
    return round(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5) / 1e5


def forecast(prices):
    """One forecast from one case's closes (oldest first, last = checkpoint)."""
    if not prices:
        raise ValueError("market has no prices")
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last):
        raise ValueError("last price is not a finite number")

    # Mean reversion over the last 240 minutes (inclusive of last).
    window = prices[-240:]
    mean240 = sum(window) / len(window)
    dev240 = math.log(last / mean240)

    # Realized volatility: sample std of log returns over each window.
    # A window of N minutes holds N log returns (N+1 closes).
    rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    v1440 = statistics.stdev(rets[-1440:]) if len(rets) >= 2 else 0.0
    v240 = statistics.stdev(rets[-240:]) if len(rets) >= 2 else 0.0

    if v240 > 0.0 and v1440 > 0.0:
        ratio = (v1440 / v240) ** P
        guess = last * math.exp(-K * dev240 * ratio)
    else:
        guess = last  # degenerate flat case: persistence
    return price5(guess)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(INPUT_DIR)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with path.open("r", encoding="utf-8") as f:
            case = json.load(f)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError(f"{path.name} does not hold exactly one market")
        price = forecast(markets[0]["prices"])
        payload = {"forecasts": [{"case": case_id, "price": price}]}
        with (out_dir / f"{case_id}.out").open("w", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
