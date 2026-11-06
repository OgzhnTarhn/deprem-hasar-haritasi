"""BRIGHT SAR uint8 ölçeği: doğrusal genlik mi, dB mi? (Faz 2 araştırması, fikir A3)

Yalnız 13 eğitim olayının zero-shot eğitim karelerinden olay başına rastgele örnek alır; Türkiye'ye dokunmaz.
Dosyalar Hugging Face zip'inden HTTP aralık isteğiyle tek tek okunur (tüm arşiv inmez).

Kullanım: uv run python scripts/arastirma/sar_olcek.py --per-event 6 --out results/arastirma/sar_olcek.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from httpzip import open_zip  # noqa: E402

from hasar.data.events import TEST_EVENT, event_of, load_standard_lists, make_zero_shot_split  # noqa: E402


def skew(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    s = x.std()
    return float(((x - x.mean()) ** 3).mean() / s**3) if s > 0 else 0.0


def block_stats(arr: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """k x k blok ortalaması ve standart sapması; içinde nodata (0) olan bloklar atılır."""
    h, w = (arr.shape[0] // k) * k, (arr.shape[1] // k) * k
    b = arr[:h, :w].astype(np.float64).reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    keep = (b > 0).all(1)
    return b[keep].mean(1), b[keep].std(1)


def speckle_slope(means: np.ndarray, stds: np.ndarray) -> float:
    """log(std) = a + b log(mean) eğimi; ortalama kutularındaki medyan std'lerle (dayanıklı).

    Çarpımsal benek (doğrusal genlik/şiddet ölçeği) b ~ 1 verir; log/dB ölçeğinde benek toplamsaldır, b ~ 0.
    """
    edges = np.quantile(means, np.linspace(0.05, 0.95, 16))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (means >= lo) & (means < hi)
        if m.sum() > 50:
            xs.append(np.log(np.median(means[m])))
            ys.append(np.log(np.median(stds[m])))
    return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-event", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    split = make_zero_shot_split(*load_standard_lists())
    rng = np.random.default_rng(a.seed)
    by_event: dict[str, list[str]] = {}
    for n in split.train:
        by_event.setdefault(event_of(n), []).append(n)
    assert TEST_EVENT not in by_event

    zf, _ = open_zip("post-event.zip")
    events = {}
    pooled = []
    bm_all, bs_all = [], []
    for ev in sorted(by_event):
        names = sorted(rng.choice(by_event[ev], size=min(a.per_event, len(by_event[ev])), replace=False).tolist())
        tiles = []
        ev_bm: list[np.ndarray] = []
        ev_bs: list[np.ndarray] = []
        for n in names:
            arr = tifffile.imread(io.BytesIO(zf.read(f"post-event/{n}_post_disaster.tif")))
            v = arr[arr > 0]
            if v.size == 0:
                tiles.append({"name": n, "valid_frac": 0.0})
                continue
            pooled.append(rng.choice(v, size=min(20000, v.size), replace=False))
            bm, bs = block_stats(arr)
            bm_all.append(bm)
            bs_all.append(bs)
            ev_bm.append(bm)
            ev_bs.append(bs)
            tiles.append(
                {
                    "name": n,
                    "valid_frac": round(float(v.size / arr.size), 4),
                    "mean": round(float(v.mean()), 2),
                    "median": float(np.median(v)),
                    "p1": float(np.percentile(v, 1)),
                    "p99": float(np.percentile(v, 99)),
                    "sat255_frac": round(float((v == 255).mean()), 5),
                    "skew": round(skew(v), 3),
                    "skew_log": round(skew(np.log(v.astype(np.float64))), 3),
                }
            )
        valid = [t for t in tiles if "skew" in t]
        events[ev] = {
            "tiles": tiles,
            "median_skew": float(np.median([t["skew"] for t in valid])) if valid else None,
            "median_skew_log": float(np.median([t["skew_log"] for t in valid])) if valid else None,
            "median_mean": float(np.median([t["mean"] for t in valid])) if valid else None,
            "speckle_slope": round(speckle_slope(np.concatenate(ev_bm), np.concatenate(ev_bs)), 3) if ev_bm else None,
        }
        print(
            ev,
            events[ev]["median_mean"],
            events[ev]["median_skew"],
            events[ev]["median_skew_log"],
            events[ev]["speckle_slope"],
            flush=True,
        )

    p = np.concatenate(pooled)
    hist = np.bincount(p, minlength=256).tolist()
    out = {
        "kind": "research",
        "question": "BRIGHT SAR uint8: dogrusal genlik mi dB mi",
        "source": "Kullervo/BRIGHT post-event.zip, zero-shot egitim kareleri, olay basina rastgele ornek",
        "seed": a.seed,
        "per_event": a.per_event,
        "pooled": {
            "n": int(p.size),
            "mean": round(float(p.mean()), 2),
            "median": float(np.median(p)),
            "skew": round(skew(p), 3),
            "skew_log": round(skew(np.log(p.astype(np.float64))), 3),
            "speckle_slope": round(speckle_slope(np.concatenate(bm_all), np.concatenate(bs_all)), 3),
            "hist": hist,
        },
        "method": (
            "8x8 blok ortalama/std; log(std)-log(ortalama) egimi: ~1 carpimsal benek (dogrusal olcek), "
            "~0 toplamsal benek (log/dB olcegi). Ek olarak ham ve log degerlerin carpikligi."
        ),
        "events": events,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        "pooled",
        out["pooled"]["mean"],
        out["pooled"]["median"],
        out["pooled"]["skew"],
        out["pooled"]["skew_log"],
        out["pooled"]["speckle_slope"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
