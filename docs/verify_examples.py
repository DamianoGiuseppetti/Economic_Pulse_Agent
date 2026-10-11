"""
Independent check of the numbers in the README examples: raw FRED REST calls and its own arithmetic,
none of the project's tools. Run from economic_pulse_agent/ with FRED_API_KEY set:

    ../.venv/bin/python ../docs/verify_examples.py

The README examples were generated on 2026-10-07 (today = "2026-10-07" below). FRED revises data and
publishes new observations, so on a later date the printed values will have moved on.
"""
import os, sys, statistics, requests
sys.path.insert(0, ".")
import config
KEY = config.FRED_API_KEY
def obs(sid, start, end, **extra):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": sid, "observation_start": start, "observation_end": end, "api_key": KEY, "file_type": "json", **extra}, timeout=30)
    r.raise_for_status()
    return {o["date"]: float(o["value"]) for o in r.json()["observations"] if o["value"] != "."}
def first_published(sid, d):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": sid, "observation_start": d,
        "observation_end": d, "output_type": 4, "realtime_start": d, "realtime_end": "9999-12-31", "api_key": KEY, "file_type": "json"}, timeout=30)
    return r.json()["observations"][0]["realtime_start"]
def next_release(rid, after):
    r = requests.get("https://api.stlouisfed.org/fred/release/dates", params={"release_id": rid, "realtime_start": after,
        "include_release_dates_with_no_data": "true", "sort_order": "asc", "limit": 1, "api_key": KEY, "file_type": "json"}, timeout=30)
    return r.json()["release_dates"][0]["date"]
def release_id(sid):
    r = requests.get("https://api.stlouisfed.org/fred/series/release", params={"series_id": sid, "api_key": KEY, "file_type": "json"}, timeout=30)
    return r.json()["releases"][0]["id"]
today = "2026-10-07"

print("== Example 1: UNRATE")
u = obs("UNRATE", "2025-01-01", today)
last = max(u); print("latest", last, u[last], "| previous", sorted(u)[-2], u[sorted(u)[-2]])
print("published", first_published("UNRATE", last), "| next release after today", next_release(release_id("UNRATE"), today))

print("== Example 2: CPIAUCSL anomalies (independent robust z)")
c = obs("CPIAUCSL", "2014-01-01", today)
dates = sorted(c); chg = {d: (c[d]/c[p]-1)*100 for p, d in zip(dates, dates[1:])}
cd = sorted(chg)
def score(d):
    lo = str(int(d[:4]) - 10) + d[4:]
    base = [chg[x] for x in cd if lo <= x < d]
    med = statistics.median(base); mad = statistics.median(abs(b-med) for b in base); sigma = 1.4826*mad
    z = (chg[d]-med)/sigma
    pct = 100*(sum(abs(b) < abs(chg[d]) for b in base) + 0.5*sum(abs(b) == abs(chg[d]) for b in base))/len(base)
    return chg[d], z, pct, len(base)
window = [d for d in cd if "2025-10-07" <= d <= today]
print("scored", len(window), "dates", window[0], "...", window[-1])
flagged = [(d, *score(d)) for d in window if abs(score(d)[1]) >= 3]
for f in flagged: print("flagged", f[0], f"change {f[1]:.4f}% z {f[2]:.2f} percentile {f[3]:.1f} baseline n {f[4]}")
print("latest", window[-1], f"{chg[window[-1]]:.3f}%", "| published", first_published("CPIAUCSL", window[-1]), "| next", next_release(release_id("CPIAUCSL"), today))

print("== Example 3: UNRATE 2025-01..2026-08 and CPILFESL")
for d in sorted(u):
    if d <= "2026-08-31": print(d[:7], u[d], end=" | ")
print()
print("UNRATE Aug 2026 published", first_published("UNRATE", "2026-08-01"), "| peak", max((v, d) for d, v in u.items() if d <= "2026-08-31"))
core = obs("CPILFESL", "2023-12-01", today)
cdts = sorted(core)
print("core Jan2025", core["2025-01-01"], "Aug2026", core["2026-08-01"], f"change {(core['2026-08-01']/core['2025-01-01']-1)*100:.2f}%")
yoy = {d: (core[d]/core[f'{int(d[:4])-1}{d[4:]}']-1)*100 for d in cdts if f'{int(d[:4])-1}{d[4:]}' in core}
for d in ("2025-01-01", "2025-08-01", "2026-05-01", "2026-08-01"): print("core YoY", d[:7], f"{yoy[d]:.2f}%")
print("core YoY max in 2026:", max((round(v,2), d) for d, v in yoy.items() if d.startswith("2026")), "| max in 2025:", max((round(v,2), d) for d, v in yoy.items() if d.startswith("2025")))
print("core Aug 2026 published", first_published("CPILFESL", "2026-08-01"), "| next", next_release(release_id("CPILFESL"), today))
