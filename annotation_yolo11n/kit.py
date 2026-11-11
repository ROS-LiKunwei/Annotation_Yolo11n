"""Portable RoboDojo HDF5 sampling and offline annotation page generation."""
import copy
import hashlib
import io
import json
import random
import shutil
from collections import Counter
from importlib.resources import files
from pathlib import Path
import h5py
import numpy as np
from PIL import Image, ImageDraw
from .schema import load_json, save_json, validate


def render(manifest):
    page = files("annotation_yolo11n").joinpath("templates/annotate.html").read_text(encoding="utf-8")
    return page.replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False).replace("</", r"<\/"))


def write_page(output, manifest):
    output = Path(output)
    save_json(output / "manifest.json", manifest)
    (output / "开始标注.html").write_text(render(manifest), encoding="utf-8")
    (output / "使用说明.txt").write_text(
        "打开开始标注.html。每个对象处理为可见点、确认不可见或忽略，全部处理后完成本图。\n"
        "每批导出JSON备份。浏览器缓存不是正式备份。低置信度模型输出保持未处理，模型候选必须人工复核。\n"
        "标注是原图像素语义中心点，不是检测框；忽略/未处理不能当作负样本。各相机分别判断。\n",
        encoding="utf-8")


def prepare(args):
    config = load_json(args.config)
    source, output = Path(args.source), Path(args.output)
    if (not config.get("cameras") or not config.get("classes")
            or config.get("frames_per_episode", 7) < 1
            or config.get("train_episode_count", 24) < 1 or config.get("val_episode_count", 6) < 1):
        raise ValueError("Cameras/classes and positive sampling counts are required")
    paths = sorted(source.glob("episode_*.hdf5"))
    if len(paths) < 3:
        raise ValueError("At least3 source episodes are required for episode-disjoint train/val/test splits")
    ids = [int(p.stem.split("_")[-1]) for p in paths]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source episode IDs")
    if args.split_policy:
        policy = load_json(args.split_policy)
        train, val, test = [list(policy[k]) for k in ["train_episodes", "val_episodes", "test_episodes"]]
        if set(train) & set(val) or set(train) & set(test) or set(val) & set(test) or set(train + val + test) != set(ids):
            raise ValueError("Split policy must partition all source episodes")
    else:
        shuffled = ids.copy()
        random.Random(config.get("seed", 20261111)).shuffle(shuffled)
        count = max(1, len(ids) // 10)
        test, val, train = sorted(shuffled[:count]), sorted(shuffled[count:2*count]), sorted(shuffled[2*count:])
        policy = dict(train_episodes=train, val_episodes=val, test_episodes=test)
    seed = config.get("seed", 20261111)
    selected_train = sorted(random.Random(seed+1).sample(train, min(len(train), config.get("train_episode_count", 24))))
    selected_val = sorted(random.Random(seed+2).sample(val, min(len(val), config.get("val_episode_count", 6))))
    selected = set(selected_train + selected_val)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episode_overviews").mkdir()
    classes, cameras = list(config["classes"]), config["cameras"]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Invalid configured class names")
    rows, episodes = [], []
    for ep, path in zip(ids, paths):
        if ep not in selected:
            continue
        with h5py.File(path, "r") as h5:
            n = len(h5[f"vision/{cameras[0]}/colors"])
            fps = float(h5["additional_info/frequency"][()])
            if n < 1 or not np.isfinite(fps) or fps <= 0:
                raise ValueError("Invalid source frame count/fps")
            if any(len(h5[f"vision/{c}/colors"]) != n for c in cameras):
                raise ValueError("Source cameras have different frame counts")
            indices = sorted({round((n-1)*k/max(1, config.get("frames_per_episode", 7)-1))
                              for k in range(config.get("frames_per_episode", 7))})
            split = "train" if ep in train else "val"
            sheet = Image.new("RGB", (320*len(cameras), 265*len(indices)), "#172536")
            draw = ImageDraw.Draw(sheet)
            for j, index in enumerate(indices):
                for ci, camera in enumerate(cameras):
                    key = f"vision/{camera}/colors"
                    raw = h5[key][index]
                    if isinstance(raw, np.ndarray) and raw.ndim == 3:
                        image = Image.fromarray(raw.astype(np.uint8), "RGB")
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=95)
                        rgb = buffer.getvalue()
                    else:
                        rgb = bytes(raw)
                        image = Image.open(io.BytesIO(rgb)).convert("RGB")
                    width, height = image.size
                    name = f"images/ep{ep:03d}_f{index:05d}_{camera}.jpg"
                    destination = output / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(rgb)
                    row = dict(id=f"{config['task']}_ep{ep:03d}_f{index:05d}_{camera}", file_name=name,
                               episode_index=ep, frame_index=index, timestamp_s=index/fps,
                               camera=camera, camera_label=config.get("camera_labels", {}).get(camera, camera),
                               width=width, height=height, split=split, rgb_sha256=hashlib.sha256(rgb).hexdigest(),
                               source_file=str(path.resolve()), source_dataset_key=key,
                               stage="未做阶段标注", selection_reasons=[f"均匀采样 {j+1}/{len(indices)}"],
                               reviewed=False, objects={c: dict(status="pending", point_xy=None) for c in classes})
                    depth = f"vision/{camera}/approximate_depths"
                    if depth in h5:
                        row["source_depth_dataset_key"] = depth
                    rows.append(row)
                    image.thumbnail((320, 240))
                    sheet.paste(image, (ci*320, j*265))
                    draw.text((ci*320+4, j*265+242), f"{camera} | f={index}", fill="white")
            sheet.save(output / f"episode_overviews/ep{ep:03d}.jpg", quality=90)
            episodes.append(dict(episode_index=ep, split=split, length_steps=n, fps=fps, selected_frames=indices))
    identity = hashlib.sha256(json.dumps([config, [(r["id"], r["rgb_sha256"], r["split"]) for r in rows]],
                                         sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest = dict(schema="robodojo_center_points_v1", task=config["task"], kit_id=identity,
                    classes=classes, class_definitions={c: {"label": v.get("label", c), "definition": v["definition"]}
                                                      for c, v in config["classes"].items()},
                    camera_order=cameras, images=rows, episodes=episodes,
                    train_episodes=selected_train, val_episodes=selected_val, protected_test_episodes=test,
                    annotation_filename_prefix=config["task"]+"_points", annotation_instructions="逐项处理每个语义中心点；不猜遮挡位置。",
                    counts=dict(images=len(rows), splits=dict(Counter(r["split"] for r in rows))))
    validate(manifest, output)
    save_json(output / "split_policy.json", policy)
    write_page(output, manifest)
    print(output / "开始标注.html")


def propose(args):
    """Predictions are unreviewed candidates, never fabricated human negatives."""
    import cv2
    from .inference import Predictor
    from .schema import image_path
    data = load_json(args.annotations)
    validate(data, args.images_root)
    if not data.get("kit_id") or not data.get("class_definitions"):
        raise ValueError("Candidate preparation needs an annotation-kit manifest with kit_id and class definitions")
    predictor = Predictor(args.checkpoint, args.bundle, args.device)
    if predictor.classes != data["classes"]:
        raise ValueError("Model/annotation class order mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = copy.deepcopy(data)
    # Reserved tests are excluded from proposal preparation as well as training.
    protected = set(data.get("protected_test_episodes", []))
    data["images"] = [r for r in data["images"] if r["split"] in {"train", "val"}
                      and r["episode_index"] not in protected]
    for row in data["images"]:
        path = image_path(args.images_root, row)
        destination = output / row["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if row["reviewed"]:
            continue
        predictions = predictor.predict(cv2.imread(str(path)), row["camera"], args.threshold)
        row["model_proposals"] = predictions
        for c, p in predictions.items():
            if row["objects"][c]["status"] == "pending":
                row["objects"][c] = dict(status="visible" if p["pred_visible"] else "pending", point_xy=p["point_xy"],
                                         source="model_proposal", confidence=p["confidence"])
        row["reviewed"] = False
    source = Path(args.images_root) / "episode_overviews"
    if source.exists():
        shutil.copytree(source, output / "episode_overviews")
    included = {r["episode_index"] for r in data["images"]}
    data["episodes"] = [e for e in data["episodes"] if e["episode_index"] in included]
    data["train_episodes"] = sorted({r["episode_index"] for r in data["images"] if r["split"] == "train"})
    data["val_episodes"] = sorted({r["episode_index"] for r in data["images"] if r["split"] == "val"})
    data["counts"] = dict(images=len(data["images"]), splits=dict(Counter(r["split"] for r in data["images"])))
    parent = data["kit_id"]
    data["parent_kit_id"] = parent
    data["root_kit_id"] = data.get("root_kit_id", parent)
    data["kit_id"] = hashlib.sha256((parent + repr([(r["id"], r.get("model_proposals")) for r in data["images"]])).encode()).hexdigest()
    data["annotation_filename_prefix"] = data.get("task", "task")+"_candidate_review"
    data["annotation_instructions"] = "模型点仅为未复核候选；低置信度仍待处理。按人工定义复核，完成后导出。"
    validate(data, output)
    write_page(output, data)
    print(output / "开始标注.html")
