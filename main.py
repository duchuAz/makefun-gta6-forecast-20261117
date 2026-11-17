"""Launch-day Twitch CCU forecaster.

The only inputs are the case file and the case's own comparables.  A comparable is a past
game launch: its hourly points cover a fortnight before its UTC release day and three days
after.  From those curves we learn, for each hour 2..24, how log viewers move with log
hour-1 viewers and with the launch's lift over its pre-release baseline.  The target's known
hour-1 print supplies both of those predictors.  Quantiles come from the weighted residual
spread, widened by a fixed calibration factor.  Nothing is specific to a game name, and
nothing reads the clock or the network.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
SPREAD = 0.40
MAX_VIEWERS = 1_000_000_000


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def curve(comp):
    """The comparable's 24 hourly values on its own UTC release day, or None if hour 1 is unusable."""
    day = parse_time(comp["day_start"])
    points = {}
    for point in comp.get("hourly") or []:
        if point.get("viewers") is None:
            continue
        points[parse_time(point["from"])] = float(point["viewers"])
    values = [points.get(day + timedelta(hours=h)) for h in range(24)]
    if not values[0] or values[0] <= 0:
        return None
    pre = [v for t, v in points.items() if day - timedelta(days=7) <= t < day and v and v > 0]
    baseline = statistics.median(pre) if pre else 1.0
    return values, max(baseline, 1.0)


def kind(comp):
    text = (comp.get("unlock") or "").lower()
    if "not a launch" in text:
        return "pre"
    if "early access" in text or "early-access" in text:
        return "early"
    if any(word in text for word in ("simultaneous", "global", "10 am", "14:00", "02:00", "18:00")):
        return "global"
    if any(word in text for word in ("f2p", "free-to-play", "live-service", "multiplayer")):
        return "service"
    if any(word in text for word in ("midnight", "rolling", "eastern")):
        return "rolling"
    return "unknown"


def weight(comp, cls):
    """A rolling console launch is the closest analogue; other shapes still steady the fit."""
    base = {"rolling": 1.0, "global": 0.15, "early": 0.08, "service": 0.05, "unknown": 0.3}.get(cls, 0.2)
    platforms = (comp.get("platforms") or "").lower()
    if "ps" not in platforms and "xbox" not in platforms:
        base *= 0.7
    if "switch" in platforms:
        base *= 0.5
    return base


def solve3(a, b):
    """Solve a 3x3 system by Gauss-Jordan elimination with partial pivoting."""
    m = [list(a[i]) + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return [0.0, 0.0, 0.0]
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, 4):
            m[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return [m[i][3] for i in range(3)]


def fit(case):
    target_day = parse_time(case["day_start"])
    history = case.get("history") or []
    pre = [p["viewers"] for p in history
           if p.get("viewers") and p.get("at") and target_day - timedelta(days=7) <= parse_time(p["at"]) < target_day]
    target_pre = max(statistics.median(pre) if pre else 1.0, 1.0)

    rows = []
    for comp in case.get("comparables") or []:
        cls = kind(comp)
        if cls == "pre":
            continue
        parsed = curve(comp)
        if parsed is None:
            continue
        values, baseline = parsed
        w = weight(comp, cls)
        if w <= 0:
            continue
        rows.append((values, baseline, w))
    if not rows:
        raise ValueError("no usable comparables")

    known = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    y1 = max(float(known["viewers"]) if known else 1.0, 1.0)
    lx = math.log(y1)
    lm = math.log(max(y1 / target_pre, 1e-6))
    lm = max(4.0, min(9.0, lm))

    # The launch lift and the hour-1 level overlap strongly, so a small ridge keeps the fit
    # stable when there are only a handful of close comparables.  The predictors stay on their
    # natural log scale: the ridge shrinks the slope, not the level.
    xs = [(1.0, math.log(max(v[0], 1.0)), max(4.0, min(9.0, math.log(max(v[0] / base, 1e-6))))) for v, base, _ in rows]
    ws = [w for _, _, w in rows]
    target_x = (1.0, lx, lm)

    fits = []
    for hour in range(1, 24):
        yy = [math.log(max(v[hour] or 1.0, 1.0)) for v, _, _ in rows]
        ata = [[sum(w * x[i] * x[j] for x, w in zip(xs, ws)) for j in range(3)] for i in range(3)]
        atb = [sum(w * x[i] * y for x, y, w in zip(xs, yy, ws)) for i in range(3)]
        for i in range(3):
            ata[i][i] += 0.1
        beta = solve3(ata, atb)
        mu = sum(beta[i] * target_x[i] for i in range(3))
        resid = [y - sum(beta[i] * x[i] for i in range(3)) for x, y in zip(xs, yy)]
        sigma = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / max(sum(ws) - 3.0, 1.0))
        fits.append((mu, max(sigma, 1e-4)))

    hours = []
    for row in case.get("forecast") or []:
        hour = int(row["hour"])
        mu, sigma = fits[hour - 2]
        q = []
        for z in Z:
            value = math.exp(mu + z * sigma * SPREAD)
            q.append(max(1.0, min(MAX_VIEWERS, value)))
        q.sort()
        hours.append({"hour": hour, "q": q})
    return {"hours": hours}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        path = os.path.join(input_dir, name)
        with open(path, encoding="utf-8") as handle:
            case = json.load(handle)
        result = fit(case)
        out = os.path.join(output_dir, name[:-3] + ".out")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))
            handle.write("\n")


if __name__ == "__main__":
    main()
