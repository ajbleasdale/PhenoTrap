# for simplyfying or renaming labels
# below is an example of simplyfying froma large number of classes to a simpler classes

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

# =========================
# CONFIG
# =========================

RENAME_RULES: Dict[str, str] = {


"background": "background",
"sky": "sky",
"deadwood": "background",
"deadwood-bare": "bare",
"water": "bare",
"bare": "bare",
"rock": "bare",

"snow-ice": "snow_ice",

"low_lying_shrub": "ericaceous_shrub",
"briar_shrub": "briar_seedling",


"graminoid": "graminoid",
"forb": "forb",
"fern": "fern",

"lichen_bryophyte": "cryptogam",
"lichen-bryophyte": "cryptogam",
"lichen": "cryptogam",
"fungus": "ignore",

"spruce": "conifer",
"pine": "conifer",
"cedar": "conifer",
"juniper": "conifer",

"birch": "broadleaf_deciduous",
"beech": "broadleaf_deciduous",
"oak": "broadleaf_deciduous",
"willow": "broadleaf_deciduous",
"acer": "broadleaf_deciduous",
"ash": "broadleaf_deciduous",
"elder": "broadleaf_deciduous",
"hawthorn": "broadleaf_deciduous",
"hazel": "broadleaf_deciduous",
"alder": "broadleaf_deciduous",
"chestnut": "broadleaf_deciduous",
"cherry": "broadleaf_deciduous",
"tree": "ignore",

"ivy": "vine",
"holly": "broadleaf_evergreen",

"leaf_litter": "bare",
"ground_cover_other": "ignore",
"ground_cover": "ignore",


"conifer_deciduous_other": "conifer",
"conifer_evergreen_other": "conifer",
"broadleaf_deciduous_other": "broadleaf_deciduous",
"broadleaf_deciduous_winter": "broadleaf_deciduous",
"broadleaf_deciduous_winter2": "ignore",
"canopy": "ignore",
"broadleaf_seedling": "briar_seedling",
"broadleaf_evergreen_other": "broadleaf_evergreen",


}


LABEL_KEYS: Set[str] = {
    "polygonlabels",
    "rectanglelabels",
    "brushlabels",
    "keypointlabels",
    "labels",
    "label",
    "choices",
    "choice",
}

IN_JSON = Path(r"\\storage-ume.slu.se\home$\arbl0002\My Documents\1-Active\Paper-1\Segmentation\JSON\FINAL-BASELINE.json")
OUT_JSON = Path(r"\\storage-ume.slu.se\home$\arbl0002\My Documents\1-Active\Paper-1\Segmentation\JSON\FINAL-BASELINE-COMPLEX.json")


def rename_label(label: str) -> str:
    return RENAME_RULES.get(label.strip(), label.strip())


changed_labels = 0
changed_lists = 0


def relabel_anywhere(obj: Any) -> None:
    global changed_labels, changed_lists

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in LABEL_KEYS and isinstance(v, list):
                new_v = [rename_label(x) if isinstance(x, str) else x for x in v]
                n_changed = sum(1 for old, new in zip(v, new_v) if old != new)
                if n_changed > 0:
                    obj[k] = new_v
                    changed_labels += n_changed
                    changed_lists += 1
            else:
                relabel_anywhere(v)

    elif isinstance(obj, list):
        for item in obj:
            relabel_anywhere(item)


with IN_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

relabel_anywhere(data)

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Input : {IN_JSON}")
print(f"Output: {OUT_JSON}")
print(f"Label values changed: {changed_labels}")
print(f"Label lists changed : {changed_lists}")
