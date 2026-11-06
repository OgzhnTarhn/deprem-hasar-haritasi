"""Bitişik binaların bağlı bileşende birleşme oranı (Faz 2 araştırması; D fikirleri, bina düzeyi oylama).

Yalnız 13 eğitim olayı (zero-shot eğitim + doğrulama kareleri). Türkiye etiketi açılmaz: hem hasar etiketleri hem
örnek etiketleri Türkiye adları süzülerek okunur.

Ölçülenler (olay başına):
- bileşen / örnek oranı: 8-komşulukta bağlı bileşen sayısının COCO örnek sayısına oranı (örnek etiketi olan olaylarda).
  1'in altı birleşmeyi gösterir.
- karışık bileşen payı: birden çok hasar sınıfı içeren bileşenlerin bina pikselleri içindeki payı. Bileşen düzeyinde
  çoğunluk oylaması bu piksellerin bir kısmını yanlış sınıfa çeker.

Kullanım:
  uv run --group arastirma python scripts/arastirma/bina_birlesme.py --target-dir <non-Türkiye target klasörü> \
      --out results/arastirma/bina_birlesme.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from httpzip import open_zip  # noqa: E402

from hasar.data.events import TEST_EVENT, event_of, load_standard_lists, make_zero_shot_split  # noqa: E402

EIGHT = np.ones((3, 3), dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    split = make_zero_shot_split(*load_standard_lists())
    names = list(split.train) + list(split.val)
    assert all(event_of(n) != TEST_EVENT for n in names)

    zf, _ = open_zip("cvprw26_trainval_instance_labels.zip")
    inst_names = {
        i.filename.split("/")[-1].replace("_instance_damage.json", ""): i
        for i in zf.infolist()
        if i.filename.endswith(".json")
    }

    ev_stats: dict[str, dict] = {}
    for n in names:
        ev = event_of(n)
        s = ev_stats.setdefault(
            ev, {"tiles": 0, "components": 0, "building_px": 0, "mixed_px": 0, "inst_tiles": 0, "inst": 0, "inst_cc": 0}
        )
        lab = tifffile.imread(a.target_dir / f"{n}_building_damage.tif")
        b = (lab > 0) & (lab < 4)
        cc, k = ndimage.label(b, structure=EIGHT)
        s["tiles"] += 1
        s["components"] += int(k)
        s["building_px"] += int(b.sum())
        if k:
            # bileşen başına farklı sınıf sayısı
            ids = cc[b]
            cls = lab[b].astype(np.int64)
            pairs = np.unique(ids * 4 + cls)
            ncls = np.bincount(pairs // 4, minlength=k + 1)
            mixed = ncls[ids] > 1
            s["mixed_px"] += int(mixed.sum())
        if n in inst_names:
            j = json.loads(zf.read(inst_names[n]).decode("utf-8-sig"))
            s["inst_tiles"] += 1
            s["inst"] += len(j["annotations"])
            s["inst_cc"] += int(k)

    events = {}
    tot = {"components": 0, "inst": 0, "inst_cc": 0, "building_px": 0, "mixed_px": 0}
    for ev in sorted(ev_stats):
        s = ev_stats[ev]
        for key in tot:
            tot[key] += s[key]
        events[ev] = {
            **s,
            "cc_per_instance": round(s["inst_cc"] / s["inst"], 3) if s["inst"] else None,
            "mixed_px_frac": round(s["mixed_px"] / s["building_px"], 4) if s["building_px"] else None,
        }
        print(ev, events[ev]["cc_per_instance"], events[ev]["mixed_px_frac"], flush=True)

    out = {
        "kind": "research",
        "question": "Bitisik binalarin bagli bilesende birlesme orani; karisik sinifli bilesen payi (13 olay)",
        "source": "BRIGHT target.zip (Turkiye haric) ve cvprw26_trainval_instance_labels.zip (Turkiye haric)",
        "connectivity": 8,
        "instance_label_events": sorted({event_of(k) for k in inst_names} - {TEST_EVENT}),
        "overall": {
            "cc_per_instance": round(tot["inst_cc"] / tot["inst"], 3) if tot["inst"] else None,
            "mixed_px_frac": round(tot["mixed_px"] / tot["building_px"], 4),
        },
        "events": events,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("overall", out["overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
