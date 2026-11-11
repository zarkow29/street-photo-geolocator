"""Build features.json: decode the 75 labelled practice photos, derive the metric and the prior.

Run:  python3 build/index_practice.py [kit-dir]
Writes features.json next to main.py. Nothing here runs at trial time.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import main as M  # noqa: E402

KIT = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/makefun/geo/photo-geolocator-01-kit"
TRIALS = (0, 1, 2)


def hav(a, b, c, d):
    p = math.radians
    dl = p(c - a)
    dn = p(d - b)
    x = math.sin(dl / 2) ** 2 + math.cos(p(a)) * math.cos(p(c)) * math.sin(dn / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(x)))


def main():
    rows = []
    for t in TRIALS:
        ref = json.load(open(f"{KIT}/practice/trial-{t}/reference/answers.json"))["photos"]
        for p in ref:
            path = f"{KIT}/practice/trial-{t}/input/env/photos/{p['id']}.jpg"
            raw = open(path, "rb").read()
            lum, cb, cr, gw, gh, mw, mh = M.decode_dc(raw)
            f = M.photo_features(lum, cb, cr, gw * 8, gh * 8, len(raw))
            assert len(f) == len(M.FEATURE_NAMES), (len(f), len(M.FEATURE_NAMES))
            rows.append({"lat": p["lat_e6"] / 1e6, "lon": p["lon_e6"] / 1e6, "f": [round(v, 5) for v in f]})
    n = len(rows)
    d = len(M.FEATURE_NAMES)
    mu = [sum(r["f"][c] for r in rows) / n for c in range(d)]
    sd = [max(1e-4, (sum((r["f"][c] - mu[c]) ** 2 for r in rows) / n) ** 0.5) for c in range(d)]

    geo = [hav(rows[i]["lat"], rows[i]["lon"], rows[j]["lat"], rows[j]["lon"])
           for i in range(n) for j in range(i + 1, n)]
    gy = sum(geo) / len(geo)
    gs = math.sqrt(sum((g - gy) ** 2 for g in geo)) or 1.0
    w = []
    for c in range(d):
        dif = [abs(rows[i]["f"][c] - rows[j]["f"][c]) for i in range(n) for j in range(i + 1, n)]
        dx = sum(dif) / len(dif)
        ds = math.sqrt(sum((v - dx) ** 2 for v in dif))
        r = abs(sum((a - dx) * (b - gy) for a, b in zip(dif, geo)) / (ds * gs)) if ds else 0.0
        w.append(r)
    wm = sum(w) / d or 1.0
    w = [max(0.03, v / wm) for v in w]

    best = None
    for la in range(-55, 71, 5):
        for lo in range(-180, 180, 5):
            s = sum(hav(r["lat"], r["lon"], la, lo) for r in rows)
            if best is None or s < best[0]:
                best = (s, float(la), float(lo))

    selfd = []
    for i, r in enumerate(rows):
        ds = sorted(math.sqrt(sum(((r["f"][c] - o["f"][c]) / sd[c] * w[c]) ** 2 for c in range(d)) / d)
                    for j, o in enumerate(rows) if j != i)
        selfd.append(ds[0])
    selfd.sort()
    threshold = round(selfd[max(0, min(n - 1, int(0.02 * n)))], 4)

    table = {
        "meta": {"source": "OpenStreetView-5M practice trials 0-2 (75 labelled photos)",
                 "built_by": "build/index_practice.py", "dims": d, "rows": n},
        "names": M.FEATURE_NAMES,
        "mu": [round(v, 5) for v in mu],
        "sd": [round(v, 5) for v in sd],
        "w": [round(v, 4) for v in w],
        "prior": [best[1], best[2]],
        "snow_mu": round(mu[M.FEATURE_NAMES.index("snow")], 5),
        "green_mu": round(mu[M.FEATURE_NAMES.index("green")], 5),
        "i_snow": M.FEATURE_NAMES.index("snow"),
        "i_green": M.FEATURE_NAMES.index("green"),
        "threshold": threshold,
        "rows": rows,
    }
    out = os.path.join(HERE, "features.json")
    with open(out, "w") as fh:
        json.dump(table, fh, separators=(",", ":"))
    print(f"wrote {out}: {n} rows x {d} dims, prior {best[1]},{best[2]}, mean prior error {best[0] / n:.0f} km")
    print(f"size {os.path.getsize(out)} bytes, threshold {threshold}")


if __name__ == "__main__":
    main()
