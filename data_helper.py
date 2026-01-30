from PIL import Image  # add this import
import os
import hashlib
from datasets import load_dataset
from PIL import Image
import json


def _is_pil_image(x):
    return isinstance(x, Image.Image)
def save_pil_to_file(pil_img, out_dir, filename_stem, ext=".jpg", quality=95):
    os.makedirs(out_dir, exist_ok=True)
    if ext.lower() in [".jpg", ".jpeg"]:
        if pil_img.mode in ("RGBA", "LA", "P"):
            pil_img = pil_img.convert("RGB")
    save_path = os.path.join(out_dir, f"{filename_stem}{ext.lower()}")
    pil_img.save(save_path, quality=quality)
    return save_path
def materialize_images_in_item(item, out_dir, item_id, prefer_ext=".jpg"):
    img = item.get("image", None)
    if _is_pil_image(img):
        stem = f"{item_id}_image"
        item["image"] = save_pil_to_file(img, out_dir, stem, ext=prefer_ext)
    elif isinstance(img, str) or img is None:
        pass
    else:
        pass
    refs = item.get("reference_frames", None)
    if refs is None:
        item["reference_frames"] = []
    elif isinstance(refs, list):
        new_refs = []
        for i, rf in enumerate(refs):
            if _is_pil_image(rf):
                stem = f"{item_id}_ref_{i:04d}"
                new_refs.append(save_pil_to_file(rf, out_dir, stem, ext=prefer_ext))
            elif isinstance(rf, str):
                new_refs.append(rf)
            else:
                continue
        item["reference_frames"] = new_refs
    else:
        item["reference_frames"] = []
    return item

def get_viper(target="data"):

    os.makedirs(target, exist_ok=True)
    json_path = os.path.join(target, "viper.json")
    if os.path.exists(json_path):
        return json_path
    images_dir = os.path.join(target, "images")
    ds = load_dataset("Monosail/VIPER", split="train")
    out = []
    for idx, ex in enumerate(ds):
        item_id = ex.get("id") or f"{idx:08d}"
        item = dict(ex)  # avoid mutating dataset internal object
        item = materialize_images_in_item(item, images_dir, item_id, prefer_ext=".jpg")
        out.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return json_path

if __name__ == "__main__":
    get_viper()