"""A1 fizibilitesi: 14 olay için Sentinel-1 SLC öncesi/sonrası çift var mı? (Faz 2 araştırması)

Olay alanı kare GeoTIFF başlıklarından gelir (13 olay: hasar etiketi dosyalarının başlığı; Türkiye: afet sonrası
SAR başlıklarından çıkarılmış küme merkezleri, `turkiye_bloklar.json`). Piksel ve etiket değeri okunmaz.
Sorgu: ASF arama API'si, Sentinel-1 SLC. Koherans için aynı yörüngede (relative orbit) olay öncesi en az 2 çekim
(öncesi-öncesi koherans) ve olay sonrası en az 1 çekim (öncesi-sonrası koherans) gerekir.

Kullanım:
  uv run --group arastirma python scripts/arastirma/s1_kapsam.py --target-dir <non-Türkiye target klasörü> \
      --turkey-json results/arastirma/turkiye_bloklar.json --out results/arastirma/s1_kapsam.json
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import tifffile
from pyproj import Transformer

# Olay tarihleri (afetin ilk günü). Kaynak: kamuya açık olay kayıtları.
DATES = {
    "bata-explosion": "2021-03-07",
    "beirut-explosion": "2020-08-04",
    "congo-volcano": "2021-05-22",
    "haiti-earthquake": "2021-08-14",
    "hawaii-wildfire": "2023-08-08",
    "la_palma-volcano": "2021-09-19",
    "libya-flood": "2023-09-10",
    "marshall-wildfire": "2021-12-30",
    "mexico-hurricane": "2023-10-25",
    "morocco-earthquake": "2023-09-08",
    "myanmar-hurricane": "2023-05-14",
    "noto-earthquake": "2024-01-01",
    "ukraine-conflict": "2022-02-24",
    "turkey-earthquake": "2023-02-06",
}
PRE_DAYS, POST_DAYS = 36, 24


def tile_center(path: Path) -> tuple[float, float]:
    with tifffile.TiffFile(path) as t:
        g = t.geotiff_metadata
        h, w = t.pages[0].shape[:2]
    sx, sy = g["ModelPixelScale"][:2]
    x0, y0 = g["ModelTiepoint"][3:5]
    cx, cy = x0 + w * sx / 2, y0 - h * sy / 2
    epsg = int(g.get("ProjectedCSTypeGeoKey") or g.get("GeographicTypeGeoKey"))
    if epsg == 4326:
        return cx, cy
    return Transformer.from_crs(epsg, 4326, always_xy=True).transform(cx, cy)


def asf(wkt: str, start: str, end: str) -> list[dict]:
    p = dict(
        platform="Sentinel-1",
        processingLevel="SLC",
        intersectsWith=wkt,
        start=start,
        end=end,
        output="geojson",
        maxResults=500,
    )
    url = "https://api.daac.asf.alaska.edu/services/search/param?" + urllib.parse.urlencode(p)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)["features"]
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ASF sorgusu başarısız: {wkt}")


def coverage(points: list[tuple[float, float]], date: str) -> dict:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    lon_c, lat_c = sum(lons) / len(lons), sum(lats) / len(lats)
    wkt = f"POINT({lon_c:.5f} {lat_c:.5f})"
    t = dt.date.fromisoformat(date)
    pre = asf(wkt, str(t - dt.timedelta(days=PRE_DAYS)), str(t))
    post = asf(wkt, str(t), str(t + dt.timedelta(days=POST_DAYS)))

    def by_track(feats):
        d = collections.defaultdict(set)
        for f in feats:
            pr = f["properties"]
            d[int(pr["pathNumber"])].add(pr["startTime"][:10])
        return d

    bp, bq = by_track(pre), by_track(post)
    tracks = []
    for tr in sorted(set(bp) & set(bq)):
        pre_d = sorted(bp[tr])
        post_d = sorted(bq[tr])
        first_post = (dt.date.fromisoformat(post_d[0]) - t).days
        tracks.append(
            {
                "track": tr,
                "pre_dates": pre_d,
                "post_dates": post_d,
                "first_post_days": first_post,
                "coherence_ok": len(pre_d) >= 2 and len(post_d) >= 1,
            }
        )
    return {
        "center_lon": round(lon_c, 5),
        "center_lat": round(lat_c, 5),
        "extent_deg": [round(min(lons), 4), round(min(lats), 4), round(max(lons), 4), round(max(lats), 4)],
        "date": date,
        "pre_scenes": len(pre),
        "post_scenes": len(post),
        "tracks": tracks,
        "feasible": any(x["coherence_ok"] for x in tracks),
        "best_first_post_days": min((x["first_post_days"] for x in tracks if x["coherence_ok"]), default=None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--turkey-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tiles", type=int, default=60, help="olay başına merkez için okunan en fazla başlık")
    a = ap.parse_args()

    files = sorted(a.target_dir.glob("*_building_damage.tif"))
    by_ev = collections.defaultdict(list)
    for f in files:
        ev = f.name.replace("_building_damage.tif", "").rsplit("_", 1)[0]
        assert ev != "turkey-earthquake"
        by_ev[ev].append(f)

    out = {
        "kind": "research",
        "question": "A1 fizibilitesi: Sentinel-1 SLC oncesi/sonrasi ciftleri (ASF)",
        "window_days": {"pre": PRE_DAYS, "post": POST_DAYS},
        "events": {},
    }
    for ev in sorted(by_ev):
        step = max(1, len(by_ev[ev]) // a.max_tiles)
        pts = [tile_center(f) for f in by_ev[ev][::step]]
        out["events"][ev] = coverage(pts, DATES[ev])
        print(ev, out["events"][ev]["feasible"], out["events"][ev]["best_first_post_days"], flush=True)

    tk = json.loads(a.turkey_json.read_text(encoding="utf-8"))
    out["events"]["turkey-earthquake"] = {"clusters": []}
    for c in tk["clusters"]:
        cov = coverage([(c["center_lon"], c["center_lat"])], DATES["turkey-earthquake"])
        cov.update(tiles=c["tiles"], place=c["nearest_place"])
        out["events"]["turkey-earthquake"]["clusters"].append(cov)
        print("turkey", c["nearest_place"], cov["feasible"], cov["best_first_post_days"], flush=True)
    t = out["events"]["turkey-earthquake"]
    t["feasible"] = all(c["feasible"] for c in t["clusters"])

    out["summary"] = {
        "feasible_events": sum(bool(v["feasible"]) for v in out["events"].values()),
        "total_events": len(out["events"]),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
