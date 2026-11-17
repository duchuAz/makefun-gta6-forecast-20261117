# GTA VI launch-day Twitch CCU forecast

A compact, deterministic forecaster for the **GTA VI launch-day Twitch viewers** competition on [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01). It reads only each case's `/input` file and writes one `/output` forecast per case.

[![make.fun competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![live rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/duchuAz/makefun-gta6-forecast-20261117/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/duchuAz/makefun-gta6-forecast-20261117/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01/leaderboard)

## How it works

The hour-1 print anchors the launch. For every comparable launch the program measures each hour 2..24 on that launch's own UTC release day, then fits a per-hour weighted ridge regression of `log(viewers)` on `log(hour-1 viewers)` and the launch lift over its pre-release baseline. Rolling-midnight console launches get the highest weight because GTA VI unlocks the same way; global, early-access and service launches are small stabilizers. The target's known hour-1 count and its pre-release median supply the predictors.

Quantiles are log-normal around each hourly fit, using the weighted residual spread and a fixed calibration factor. The output keeps the exact 23-row, five-quantile shape the grader requires and clamps every value to `(0, 1e9]`.

## Run locally

```sh
python3 main.py
```

Set `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` to use directories other than `/input` and `/output`. The program is Python-standard-library only, deterministic, and silent on success.

## Files

- `main.py` — the entrypoint and the whole model.
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout-wallet claim for the repository.

## Verification

The entry was checked with the competition's own `makefun check`, including the transcript gate, before submission. Practice outputs were compared with the published practice actuals; the program never reads those actuals, the network, or anything outside its input file.
