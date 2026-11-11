"""Validate original-pixel labels, review state, identity hashes, and episode isolation."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

STATUSES = {"visible", "not_visible", "ignore", "pending"}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def image_path(root, row):
    root = Path(root).resolve()
    path = (root / row["file_name"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image escapes images root: {row['id']}")
    return path


def validate(data, images_root=None, for_training=False):
    classes = data["classes"]
    if not classes or len(classes) != len(set(classes)) or not all(isinstance(c, str) and c for c in classes):
        raise ValueError("Classes must be distinct nonempty strings")
    if for_training and data.get("training_allowed") is False:
        raise ValueError("This export is evaluation/review only, not a training dataset")
    rows = data["images"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty dataset or duplicate image IDs")
    episode_splits = {}
    protected = set(data.get("protected_test_episodes", []))
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        ep = row["episode_index"]
        if ep in episode_splits and episode_splits[ep] != row["split"]:
            raise ValueError(f"Episode {ep} crosses data splits")
        episode_splits[ep] = row["split"]
        if ep in protected and row["split"] != "test":
            raise ValueError(f"Protected episode {ep} entered train/val")
        if type(row["reviewed"]) is not bool or set(row["objects"]) != set(classes):
            raise ValueError(f"Invalid review state or object keys: {row['id']}")
        if any(type(row[k]) is not int or row[k] <= 0 for k in ["width", "height"]):
            raise ValueError("Invalid image dimensions")
        for obj in row["objects"].values():
            status, point = obj["status"], obj["point_xy"]
            if status not in STATUSES or (row["reviewed"] and status == "pending"):
                raise ValueError(f"Invalid/unprocessed reviewed label: {row['id']}")
            if status == "visible":
                if (not isinstance(point, list) or len(point) != 2
                        or not all(type(v) in (int, float) and math.isfinite(v) for v in point)
                        or not 0 <= point[0] < row["width"] or not 0 <= point[1] < row["height"]):
                    raise ValueError(f"Invalid visible coordinate: {row['id']}")
            elif point is not None:
                raise ValueError("Non-visible/unknown labels must have null coordinates")
        if images_root is not None:
            path = image_path(images_root, row)
            if row.get("rgb_sha256") and digest(path) != row["rgb_sha256"]:
                raise ValueError(f"Image SHA256 mismatch: {row['id']}")
            from PIL import Image
            with Image.open(path) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError(f"Image dimensions differ: {row['id']}")
    if for_training:
        tr = {r.get("rgb_sha256") for r in rows if r["reviewed"] and r["split"] == "train"} - {None}
        va = {r.get("rgb_sha256") for r in rows if r["reviewed"] and r["split"] == "val"} - {None}
        if tr & va:
            raise ValueError("Identical RGB hashes cross train/validation splits")
    return {"images": len(rows), "reviewed": sum(r["reviewed"] for r in rows),
            "by_split": dict(Counter(r["split"] for r in rows)),
            "statuses": dict(Counter(o["status"] for r in rows for o in r["objects"].values())),
            "episode_isolation": True}
