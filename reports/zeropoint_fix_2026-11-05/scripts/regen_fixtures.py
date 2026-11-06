"""Regenerate golden flux values for the corrected zeropoint direction.
Correct FLUXCAL (zp 27.5) needs no magzpsci at all:
    flux     = sign * 10**(-0.4*(magpsf - 27.5))
    flux_err = |flux| * ln(10)/2.5 * sigmapsf
    upper-limit flux_err = 10**(-0.4*(diffmaglim - 27.5)) / 5
Old (inverted) value, used only to assert we're editing the right rows:
    old = raw * 10**((zp - 27.5)/2.5),  raw = 10**(-0.4*(m - zp))"""
import math, re, importlib.util, sys
Z = 27.5; K = math.log(10) / 2.5
def load(p):
    s = importlib.util.spec_from_file_location("m", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def old(m, zp): return 10 ** (-0.4 * (m - zp)) * 10 ** ((zp - Z) / 2.5)
def new(m): return 10 ** (-0.4 * (m - Z))
def close(a, b): return abs(a - b) <= 1e-9 * max(1, abs(b))
def fmt(xs): return "[\n" + "".join(f"    {x!r},\n" for x in xs) + "]"

# ---- BTSbot
p = "tests/fixtures/golden_btsbot_object.py"; f = load(p); src = open(p).read()
for raw_name, flux_name, err_name in [("RAW_ROWS", "EXPECTED_FLUX", "EXPECTED_FLUX_ERR"),
                                      ("SENTINEL_RAW_ROWS", "SENTINEL_EXPECTED_FLUX", "SENTINEL_EXPECTED_FLUX_ERR")]:
    rows = getattr(f, raw_name); oldf = getattr(f, flux_name); olde = getattr(f, err_name)
    nf, ne = [], []
    for (jd, fid, m, s, zp, pos), of, oe in zip(rows, oldf, olde):
        sign = 1 if pos else -1
        assert close(sign * old(m, zp), of), (raw_name, m, zp, of)
        assert close(abs(of) * K * s, oe), (raw_name, "err")
        nf.append(sign * new(m)); ne.append(abs(nf[-1]) * K * s)
    src = re.sub(rf"^{flux_name} = \[.*?^\]", f"{flux_name} = " + fmt(nf), src, count=1, flags=re.S | re.M)
    src = re.sub(rf"^{err_name} = \[.*?^\]", f"{err_name} = " + fmt(ne), src, count=1, flags=re.S | re.M)
    print(raw_name, "first row old->new flux:", oldf[0], "->", nf[0])
open(p, "w").write(src)

# ---- avro ZTF
p = "tests/fixtures/golden_avro_ztf_alert.py"; f = load(p); src = open(p).read()
rows = sorted([f.CANDIDATE] + list(f.PRV_CANDIDATES), key=lambda r: r["jd"])
out = []
for r, (mjd, fid, of, oe, det) in zip(rows, f.EXPECTED_ROWS):
    assert close(r["jd"] - 2400000.5, mjd) and r["fid"] == fid
    if r["magpsf"] is None:
        assert of == 0.0 and close(old(r["diffmaglim"], r["magzpsci"]) / 5.0, oe)
        nf, ne = 0.0, new(r["diffmaglim"]) / 5.0
    else:
        sign = 1 if r["isdiffpos"] == "t" else -1
        assert close(sign * old(r["magpsf"], r["magzpsci"]), of) and close(abs(of) * K * r["sigmapsf"], oe)
        nf = sign * new(r["magpsf"]); ne = abs(nf) * K * r["sigmapsf"]
    out.append((mjd, fid, nf, ne, det))
body = "EXPECTED_ROWS = [\n" + "".join(f"    {t!r},\n" for t in out) + "]"
src = re.sub(r"^EXPECTED_ROWS = \[.*?^\]", body, src, count=1, flags=re.S | re.M)
open(p, "w").write(src)
print("ztf candidate old->new flux:", f.EXPECTED_ROWS[-1][2], "->", out[-1][2])
print("all old values matched the inverted formula; fixtures rewritten")
