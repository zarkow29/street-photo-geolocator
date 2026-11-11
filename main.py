"""Street-photo geolocator: a model-free baseline-JPEG DC decoder and a neighbour prior.

Reads the trial env only. For every photo it Huffman-decodes the entropy-coded scan just far
enough to keep each block's DC coefficient (luma at 1/8 scale, chroma means), derives coarse
scene statistics from that grid, and places the photo with a k-nearest-neighbour vote over the
labelled practice features, shrunk toward a global prior. No model call, no clock, no unseeded
randomness.
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
import makefun_agent as af  # noqa: E402  (vendored trial shim: read_trial, trial_rng)

# Wall budget for fetching and decoding; past it the remaining photos take the prior.
FETCH_BUDGET_S = 420.0
K = 5
NEIGHBOUR_WEIGHT = 0.15
PHOTO_TIMEOUT_S = 45.0


def _extend(v, s):
    return v - (1 << s) + 1 if v < (1 << (s - 1)) else v


def _huff(bits, vals):
    """Canonical Huffman decoder tables: maxcode, mincode, valptr."""
    maxcode = [-1] * 18
    mincode = [0] * 18
    valptr = [0] * 18
    code = 0
    k = 0
    for l in range(1, 17):
        n = bits[l - 1]
        valptr[l] = k
        mincode[l] = code
        code += n
        k += n
        maxcode[l] = code - 1 if n else -1
        code <<= 1
    maxcode[17] = 1 << 30
    return maxcode, mincode, valptr, vals


def _parse(b):
    """Segments up to SOS: quantisers, Huffman tables, frame; then the cleaned scan bytes."""
    qt = {}
    hts = {}
    frame = None
    scan = None
    i = 2
    n = len(b)
    while i < n - 1:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0xD9, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        L = (b[i + 2] << 8) | b[i + 3]
        seg = b[i + 4:i + 2 + L]
        if m == 0xDB:
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tq = seg[j] & 15
                j += 1
                if pq:
                    qt[tq] = [((seg[j + 2 * k] << 8) | seg[j + 2 * k + 1]) for k in range(64)]
                    j += 128
                else:
                    qt[tq] = list(seg[j:j + 64])
                    j += 64
        elif m == 0xC4:
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                th = seg[j] & 15
                j += 1
                bits = list(seg[j:j + 16])
                j += 16
                cnt = sum(bits)
                vals = list(seg[j:j + cnt])
                j += cnt
                hts[(tc, th)] = _huff(bits, vals)
        elif m == 0xC0:
            w = (seg[3] << 8) | seg[4]
            h = (seg[1] << 8) | seg[2]
            comps = []
            for c in range(seg[5]):
                hv = seg[7 + c * 3]
                comps.append((seg[6 + c * 3], hv >> 4, hv & 15, seg[8 + c * 3]))
            frame = (w, h, comps)
        elif m == 0xDA:
            scan = [(seg[1 + c * 2], seg[2 + c * 2] >> 4, seg[2 + c * 2] & 15) for c in range(seg[0])]
            i += 2 + L
            break
        i += 2 + L
    if frame is None or scan is None:
        raise ValueError("not a baseline JPEG")
    out = bytearray()
    restarts = []
    while i < n - 1:
        c = b[i]
        if c == 0xFF:
            m2 = b[i + 1]
            if m2 == 0x00:
                out.append(0xFF)
            elif 0xD0 <= m2 <= 0xD7:
                restarts.append(len(out))
            else:
                break
            i += 2
            continue
        out.append(c)
        i += 1
    return frame, scan, hts, qt, bytes(out), restarts


def decode_dc(b):
    """Decode DC coefficients only: (luma grid, cb grid, cr grid, gw, gh, mw, mh)."""
    (w, h, comps), scan, hts, qt, data, restarts = _parse(b)
    hmax = max(c[1] for c in comps)
    vmax = max(c[2] for c in comps)
    mw = (w + 8 * hmax - 1) // (8 * hmax)
    mh = (h + 8 * vmax - 1) // (8 * vmax)
    gw, gh = (w + 7) // 8, (h + 7) // 8
    lum = [[0] * gw for _ in range(gh)]
    cb = [[0] * mw for _ in range(mh)]
    cr = [[0] * mw for _ in range(mh)]
    dc_tabs = [hts[(0, s[1])] for s in scan]
    ac_tabs = [hts[(1, s[2])] for s in scan]
    qd = [qt.get(c[3], [1])[0] for c in comps]
    pred = [0] * len(comps)
    pos = 0
    acc = 0
    nb = 0
    ln = len(data)

    def fill():
        nonlocal pos, acc, nb
        while nb <= 24:
            c = data[pos] if pos < ln else 0
            pos += 1
            acc = ((acc << 8) | c) & 0xFFFFFFFFFFFF
            nb += 8

    def sym(tab):
        nonlocal nb
        mc, mnc, vp, vals = tab
        fill()
        peek = (acc >> (nb - 16)) & 0xFFFF
        l = 1
        code = peek >> 15
        while l < 16 and code > mc[l]:
            l += 1
            code = peek >> (16 - l)
        v = vals[vp[l] + code - mnc[l]]
        nb -= l
        return v

    for my in range(mh):
        base = my * mw
        for mx in range(mw):
            if restarts and base + mx and (base + mx) % len(restarts) == 0:
                pred = [0] * len(comps)
                pos = restarts[(base + mx) // len(restarts) - 1]
                acc = 0
                nb = 0
            for ci, comp in enumerate(comps):
                dct, act = dc_tabs[ci], ac_tabs[ci]
                for by in range(comp[2]):
                    for bx in range(comp[1]):
                        s = sym(dct)
                        if s:
                            fill()
                            pred[ci] += _extend((acc >> (nb - s)) & ((1 << s) - 1), s)
                            nb -= s
                        if ci == 0:
                            yy = my * comp[2] + by
                            xx = mx * comp[1] + bx
                            if yy < gh and xx < gw:
                                lum[yy][xx] = pred[0] * qd[0]
                        k = 1
                        while k < 64:
                            rs = sym(act)
                            r = rs >> 4
                            s2 = rs & 15
                            if s2 == 0:
                                if r == 15:
                                    k += 16
                                    continue
                                break
                            k += r
                            fill()
                            nb -= s2
                            k += 1
                if ci == 1:
                    cb[my][mx] = pred[1] * qd[1] / max(1, comp[1] * comp[2])
                elif ci == 2:
                    cr[my][mx] = pred[2] * qd[2] / max(1, comp[1] * comp[2])
    return lum, cb, cr, gw, gh, mw, mh


# === FEATURES ===
FEATURE_NAMES = ["lum", "lstd", "sat", "sky", "skytop", "green", "sand", "grey", "snow",
                 "water", "horizon", "rowcon", "tex", "bpp", "aspect", "p10", "p90",
                 "cb_top", "cb_gap", "green_top", "green_gap", "lum_top", "lum_bot", "ent",
                 "mcb", "mcr", "cbs", "crs", "abcb", "abcr"] + [f"cb{i}" for i in range(16)] \
    + [f"cr{i}" for i in range(16)] + [f"lh{i}" for i in range(8)] + [f"t{i}" for i in range(27)]


def photo_features(lum, cb, cr, w, h, nbytes):
    """Scene statistics from the DC grid: climate shares, layout, colour histograms."""
    gh = len(lum)
    gw = len(lum[0])
    mh = len(cb)
    mw = len(cb[0])
    L = [[v / 8 + 128 for v in r] for r in lum]
    rx, ry = max(1, gw // mw), max(1, gh // mh)
    CB = [[cb[min(mh - 1, i // ry)][min(mw - 1, j // rx)] / 8 + 128 for j in range(gw)] for i in range(gh)]
    CR = [[cr[min(mh - 1, i // ry)][min(mw - 1, j // rx)] / 8 + 128 for j in range(gw)] for i in range(gh)]
    n = gh * gw
    flat = [v for r in L for v in r]
    mean = sum(flat) / n
    lstd = math.sqrt(sum((v - mean) ** 2 for v in flat) / n)
    sat = sum(abs(CB[i][j] - 128) + abs(CR[i][j] - 128) for i in range(gh) for j in range(gw)) / (2 * n)
    sky = green = sand = grey = snow = water = 0
    for i in range(gh):
        for j in range(gw):
            v = L[i][j]
            b = CB[i][j] - 128
            r = CR[i][j] - 128
            if v > 150 and b > 4 and r < 4:
                sky += 1
            if r < -6 and b < -4:
                green += 1
            if r > 6 and b < -2 and v > 110:
                sand += 1
            if abs(b) < 6 and abs(r) < 6 and v > 60:
                grey += 1
            if v > 205 and abs(b) + abs(r) < 10:
                snow += 1
            if v < 90 and b > 7:
                water += 1
    h3 = max(1, gh // 3)
    top, bot = L[:h3], L[-h3:]
    skytop = sum(1 for r in top for v in r if v > 150) / (h3 * gw)
    rows = [sum(r) / gw for r in L]
    jumb = 0.0
    hz = 0.5
    for i in range(1, len(rows)):
        d = abs(rows[i] - rows[i - 1])
        if d > jumb:
            jumb = d
            hz = i / len(rows)
    tex = 0.0
    for i in range(gh):
        for j in range(gw):
            v = L[i][j]
            if j + 1 < gw:
                tex += abs(v - L[i][j + 1])
            if i + 1 < gh:
                tex += abs(v - L[i + 1][j])
    tex /= (2 * n) * 255.0
    srt = sorted(flat)
    p10 = srt[n // 10] / 255.0
    p90 = srt[9 * n // 10] / 255.0

    def m(rows_):
        return sum(sum(r) for r in rows_) / max(1, len(rows_) * gw)

    cb_top = sum(sum(r) for r in CB[:h3]) / (h3 * gw) - 128
    cb_bot = sum(sum(r) for r in CB[-h3:]) / (h3 * gw) - 128
    g_top = sum(1 for i in range(h3) for j in range(gw) if CR[i][j] < 122) / (h3 * gw)
    g_bot = sum(1 for i in range(gh - h3, gh) for j in range(gw) if CR[i][j] < 122) / (h3 * gw)
    hist = [0] * 8
    for v in flat:
        hist[min(7, int(v / 32))] += 1
    ent = -sum((c / n) * math.log(c / n + 1e-9) for c in hist) / math.log(8)
    cbh = [0] * 16
    crh = [0] * 16
    for i in range(gh):
        for j in range(gw):
            cbh[min(15, max(0, int((CB[i][j] - 96) / 4)))] += 1
            crh[min(15, max(0, int((CR[i][j] - 96) / 4)))] += 1
    cbs = math.sqrt(sum((CB[i][j] - 128) ** 2 for i in range(gh) for j in range(gw)) / n)
    crs = math.sqrt(sum((CR[i][j] - 128) ** 2 for i in range(gh) for j in range(gw)) / n)
    tiles = []
    for ty in range(3):
        for tx in range(3):
            ys = range(ty * gh // 3, max(ty * gh // 3 + 1, (ty + 1) * gh // 3))
            xs = range(tx * gw // 3, max(tx * gw // 3 + 1, (tx + 1) * gw // 3))
            k = 0
            sl = sb = sr = 0.0
            for i in ys:
                for j in xs:
                    sl += L[i][j]
                    sb += CB[i][j] - 128
                    sr += CR[i][j] - 128
                    k += 1
            tiles += [sl / k / 255.0, sb / k / 40.0, sr / k / 40.0]
    return [mean, lstd, sat, sky / n, skytop, green / n, sand / n, grey / n, snow / n,
            water / n, hz, m(bot) - m(top), tex, nbytes / (w * h), math.log(max(1, w) / max(1, h)),
            p10, p90, cb_top, cb_top - cb_bot, g_top, g_top - g_bot, m(top), m(bot), ent,
            sum(sum(r) for r in CB) / n - 128, sum(sum(r) for r in CR) / n - 128,
            cbs, crs,
            sum(abs(CB[i][j] - 128) for i in range(gh) for j in range(gw)) / n,
            sum(abs(CR[i][j] - 128) for i in range(gh) for j in range(gw)) / n] \
        + [c / n for c in cbh] + [c / n for c in crh] + [c / n for c in hist] + tiles


# === RUNTIME ===


def http_get(url, timeout=60.0):
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read()


def http_post_json(url, obj, timeout=60.0):
    body = json.dumps(obj, separators=(",", ":")).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def prior_answer(table):
    return clamp(table["prior"][0], -85.0, 85.0), clamp(table["prior"][1], -179.9, 179.9)


def place(feats, table, jitter):
    """Prior nudged by the two climate cues, then shrunk toward a confident neighbour vote."""
    la = table["prior"][0] - 60.0 * (feats[table["i_snow"]] - table["snow_mu"]) \
        + 30.0 * (feats[table["i_green"]] - table["green_mu"])
    la = clamp(la, -85.0, 85.0)
    lo = table["prior"][1]
    sd = table["sd"]
    w = table["w"]
    ndim = len(table["names"])
    best = []
    for row, jit in zip(table["rows"], jitter):
        f = row["f"]
        s = 0.0
        for c in range(ndim):
            d = (feats[c] - f[c]) / sd[c] * w[c]
            s += d * d
        best.append((s + jit, row))
    best.sort(key=lambda x: x[0])
    near = math.sqrt(best[0][0] / ndim)
    if near <= table["threshold"] * 0.1:
        return clamp(best[0][1]["lat"], -90.0, 90.0), clamp(best[0][1]["lon"], -180.0, 180.0)
    if near <= table["threshold"]:
        nb = best[:K]
        nla = sorted(r["lat"] for _, r in nb)[K // 2]
        nlo = sorted(r["lon"] for _, r in nb)[K // 2]
        la = la * (1.0 - NEIGHBOUR_WEIGHT) + nla * NEIGHBOUR_WEIGHT
        lo = lo * (1.0 - NEIGHBOUR_WEIGHT) + nlo * NEIGHBOUR_WEIGHT
    return clamp(la, -90.0, 90.0), clamp(lo, -180.0, 180.0)


def main():
    started = time.monotonic()
    trial = af.read_trial()
    rng = af.trial_rng(trial, "neighbour-tiebreak")
    with open(os.path.join(HERE, "features.json")) as fh:
        table = json.load(fh)
    jitter = [rng.random() * 1e-9 for _ in table["rows"]]

    listing = json.loads(http_get(trial.env_url + "/photos", timeout=60.0))
    photos = listing["photos"]
    answers = {}
    pl, po = prior_answer(table)
    for p in photos:
        pid = p["id"]
        if time.monotonic() - started > FETCH_BUDGET_S:
            answers[pid] = (pl, po)
            continue
        try:
            url = urllib.parse.urljoin(trial.env_url + "/photos", p.get("url") or f"/photos/{pid}.jpg")
            raw = http_get(url, timeout=PHOTO_TIMEOUT_S)
            lum, cb, cr, gw, gh, mw, mh = decode_dc(raw)
            f = photo_features(lum, cb, cr, gw * 8, gh * 8, len(raw))
            if len(f) != len(feature_names := table["names"]):
                raise ValueError("feature length")
            answers[pid] = place(f, table, jitter)
        except Exception:
            answers[pid] = (pl, po)

    payload = [{"id": pid, "lat": round(la, 6), "lon": round(lo, 6)} for pid, (la, lo) in answers.items()]
    status, body = http_post_json(trial.env_url + "/answers", {"answers": payload})
    if status == 400 and payload:
        text = body.decode("utf-8", "replace")
        bad = None
        for pid in answers:
            if pid in text:
                bad = pid
                break
        if bad is None:
            bad = payload[0]["id"]
        retry = [a for a in payload if a["id"] != bad]
        if retry:
            http_post_json(trial.env_url + "/answers", {"answers": retry})


if __name__ == "__main__":
    main()


