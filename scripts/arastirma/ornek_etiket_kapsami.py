"""BRIGHT bina örnek etiketlerinin (CVPRW26 trainval) hangi karelerde olduğu (Faz 2 araştırması, protokol §7).

YALNIZ dosya adları: zip'in merkez dizini HTTP aralık isteğiyle okunur; hiçbir JSON açılmaz. Türkiye için yalnız
"kaç karede örnek etiketi var" sayılır.

Kullanım: uv run python scripts/arastirma/ornek_etiket_kapsami.py --out results/arastirma/ornek_etiket_kapsami.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from httpzip import open_zip  # noqa: E402

from hasar.data.events import TEST_EVENT, event_of, load_standard_lists, make_zero_shot_split  # noqa: E402

SUFFIX = "_instance_damage.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    zf, _ = open_zip("cvprw26_trainval_instance_labels.zip")
    have = {i.filename.split("/")[-1][: -len(SUFFIX)] for i in zf.infolist() if i.filename.endswith(SUFFIX)}
    split = make_zero_shot_split(*load_standard_lists())

    def cover(names):
        c_all = Counter(event_of(n) for n in names)
        c_have = Counter(event_of(n) for n in names if n in have)
        return {ev: {"tiles": c_all[ev], "with_instances": c_have[ev]} for ev in sorted(c_all)}

    out = {
        "kind": "research",
        "question": "Bina ornek etiketleri hangi karelerde var (yalniz dosya adlari)",
        "source": "Kullervo/BRIGHT cvprw26_trainval_instance_labels.zip merkez dizini",
        "format": "COCO benzeri JSON: images[].geo_reference, annotations[].segmentation/bbox/category_id",
        "train": cover(split.train),
        "val": cover(split.val),
        "test": {TEST_EVENT: {"tiles": len(split.test), "with_instances": sum(n in have for n in split.test)}},
        "unlisted_names": len(have - set(split.train) - set(split.val) - set(split.test)),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["test"]), out["unlisted_names"])
    for part in ("train", "val"):
        print(part, {k: v["with_instances"] for k, v in out[part].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
