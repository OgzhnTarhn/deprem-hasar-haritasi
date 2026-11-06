"""Türkiye karelerinin mekânsal dağılımı (Faz 2 araştırması; bootstrap birimi için).

YALNIZ meta veri: her Türkiye afet sonrası SAR karesinin GeoTIFF başlığındaki konum etiketleri okunur. Piksel
değeri çözülmez, etiket dosyası açılmaz. Zip üyesinin yalnız ilk birkaç KB'ı HTTP aralık isteğiyle indirilir
(BRIGHT TIFF'lerinde IFD dosyanın 8. baytında başlar); geri kalanı sıfırla doldurulur ve hiç okunmaz.

Kullanım:
  uv run --group arastirma python scripts/arastirma/turkiye_bloklar.py --out results/arastirma/turkiye_bloklar.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from httpzip import BASE, HttpFile  # noqa: E402

from hasar.data.events import tile_stem  # noqa: E402
from hasar.guard import is_turkey_tile  # noqa: E402

HEAD_BYTES = 16384

# Yaklaşık ilçe/şehir merkezleri (enlem, boylam); yalnız kümelere ad vermek için.
PLACES = {
    "Kahramanmaraş": (37.585, 36.937),
    "Antakya": (36.202, 36.160),
    "İskenderun": (36.587, 36.173),
    "Gaziantep": (37.066, 37.383),
    "Nurdağı": (37.176, 36.738),
    "İslahiye": (37.026, 36.631),
    "Adıyaman": (37.764, 38.276),
    "Malatya": (38.355, 38.309),
    "Osmaniye": (37.074, 36.247),
    "Kırıkhan": (36.497, 36.358),
    "Türkoğlu": (37.383, 36.850),
    "Elbistan": (38.205, 37.198),
    "Pazarcık": (37.487, 37.290),
    "Samandağ": (36.084, 35.977),
    "Hassa": (36.799, 36.518),
    "Gölbaşı": (37.783, 37.640),
    "Diyarbakır": (37.914, 40.230),
    "Şanlıurfa": (37.159, 38.796),
    "Adana": (37.000, 35.321),
    "Defne": (36.160, 36.130),
    "Reyhanlı": (36.268, 36.567),
}


def head_meta(zf, info):
    """Read only the member's first bytes, pad to full size, parse tags (pixels never decoded)."""
    with zf.open(info) as m:
        head = m.read(HEAD_BYTES)
    buf = io.BytesIO(head + b"\0" * (info.file_size - len(head)))
    with tifffile.TiffFile(buf) as t:
        g = t.geotiff_metadata
        h, w = t.pages[0].shape[:2]
    sx, sy = g["ModelPixelScale"][:2]
    x0, y0 = g["ModelTiepoint"][3:5]
    epsg = int(g.get("ProjectedCSTypeGeoKey") or g.get("GeographicTypeGeoKey"))
    return epsg, x0 + w * sx / 2, y0 - h * sy / 2, w * sx, h * sy


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    d = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--link-km", type=float, default=1.0, help="tek bağlantı kümeleme eşiği")
    ap.add_argument("--block-km", type=float, default=2.0, help="ızgara blok boyu")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    import zipfile

    def opener():
        f = HttpFile(BASE + "post-event.zip")
        return zipfile.ZipFile(io.BufferedReader(f, buffer_size=HEAD_BYTES))

    zf0 = opener()
    infos = [
        i
        for i in zf0.infolist()
        if i.filename.endswith(".tif") and is_turkey_tile(tile_stem(i.filename.split("/")[-1]))
    ]
    print("turkey tiles:", len(infos), flush=True)

    chunks = [infos[k :: a.threads] for k in range(a.threads)]

    def work(chunk):
        zf = opener()
        return [(tile_stem(i.filename.split("/")[-1]), *head_meta(zf, i)) for i in chunk]

    with ThreadPoolExecutor(a.threads) as ex:
        rows = [r for part in ex.map(work, chunks) for r in part]
    rows.sort()

    latlon = []
    for _name, epsg, cx, cy, _wx, _wy in rows:
        if epsg == 4326:
            latlon.append((cy, cx))
        else:
            lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(cx, cy)
            latlon.append((lat, lon))
    ll = np.array(latlon)
    epsgs = sorted({r[1] for r in rows})

    # Tek bağlantı kümeleme (birleşim-bul)
    n = len(ll)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = haversine_km(ll[i][None, :], ll)
        for j in np.nonzero(d < a.link_km)[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj
    roots = [find(i) for i in range(n)]
    clusters = {}
    for i, r in enumerate(roots):
        clusters.setdefault(r, []).append(i)

    names_pl = list(PLACES)
    pl = np.array([PLACES[k] for k in names_pl])
    out_clusters = []
    for idx in sorted(clusters.values(), key=len, reverse=True):
        c = ll[idx].mean(0)
        d = haversine_km(c[None, :], pl)
        k = int(d.argmin())
        out_clusters.append(
            {
                "tiles": len(idx),
                "center_lat": round(float(c[0]), 4),
                "center_lon": round(float(c[1]), 4),
                "nearest_place": names_pl[k],
                "nearest_km": round(float(d[k]), 1),
            }
        )

    # Izgara blokları (yaklaşık km; enlem başına 111 km)
    lat0 = ll[:, 0].mean()
    gy = np.floor(ll[:, 0] * 111.0 / a.block_km).astype(int)
    gx = np.floor(ll[:, 1] * 111.0 * np.cos(np.radians(lat0)) / a.block_km).astype(int)
    blocks = {}
    for y, x in zip(gy, gx, strict=True):
        blocks[(y, x)] = blocks.get((y, x), 0) + 1
    sizes = sorted(blocks.values(), reverse=True)

    out = {
        "kind": "research",
        "question": "Turkiye karelerinin sehir ve mekansal blok dagilimi (yalniz GeoTIFF basligi)",
        "source": "Kullervo/BRIGHT post-event.zip, Turkiye karelerinin TIFF basliklari; piksel ve etiket okunmadi",
        "tiles": n,
        "crs_epsg": epsgs,
        "tile_size_m_median": [round(float(np.median([r[4] for r in rows if r[1] != 4326] or [0])), 1)],
        "link_km": a.link_km,
        "clusters": out_clusters,
        "block_km": a.block_km,
        "blocks": {"count": len(sizes), "tiles_per_block": sizes},
        "tile_centers": [
            {"name": r[0], "lat": round(float(p[0]), 6), "lon": round(float(p[1]), 6)}
            for r, p in zip(rows, ll, strict=True)
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for c in out_clusters:
        print(c)
    print("blocks:", len(sizes), sizes[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
