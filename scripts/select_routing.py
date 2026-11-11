#!/usr/bin/env python3
"""Select per-camera spatial checkpoints and optional existing YOLO visibility.

Development-only selection: no model gradients or label editing are performed.
The optional visibility bundle uses the published R3 camera_routing.json format.
"""
import argparse
import copy
import math
import shutil
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from annotation_yolo11n.inference import Predictor
from annotation_yolo11n.metrics import point_metrics
from annotation_yolo11n.schema import load_json, save_json, validate, image_path, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--visibility-bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--error-px", type=float, default=10.)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = load_json(args.annotations)
    validate(data, args.images_root)
    rows = [r for r in data["images"] if r["reviewed"] and r["split"] == "val"]
    if not rows:
        raise ValueError("Reviewed validation rows required")
    classes = data["classes"]
    cameras = sorted({r["camera"] for r in rows})
    old_scores, old_paths = {}, {}
    if args.visibility_bundle:
        old = load_json(args.visibility_bundle / "camera_routing.json")
        if old["classes"] != classes:
            raise ValueError("Visibility class order mismatch")
        for cam in cameras:
            spec = old["cameras"][cam]
            path = args.visibility_bundle / spec["weights"]
            if spec.get("sha256") and digest(path) != spec["sha256"]:
                raise ValueError("Visibility weight hash mismatch")
            old_paths[cam] = path
            model = YOLO(str(path))
            for row in [r for r in rows if r["camera"] == cam]:
                res = model.predict(str(image_path(args.images_root, row)), imgsz=640, rect=False,
                                    conf=.01, max_det=1, device=args.device, verbose=False)[0]
                scores = res.keypoints.data[0, :, 2].cpu().tolist() if len(res.boxes) else [0.] * len(classes)
                if len(scores) != len(classes):
                    raise ValueError("Visibility keypoint count mismatch")
                old_scores[row["id"]] = dict(zip(classes, scores))
    choices, picked, all_metrics = {}, {}, {}
    input_size = None
    for checkpoint in args.checkpoints:
        model = Predictor(checkpoint=checkpoint, device=args.device)
        if model.classes != classes or (input_size and input_size != model.input_size):
            raise ValueError("Candidate class/input configurations differ")
        input_size = model.input_size
        records = []
        for row in rows:
            xy, conf, _ = model.predict_arrays(cv2.imread(str(image_path(args.images_root, row))))
            for j, name in enumerate(classes):
                obj = row["objects"][name]
                records.append(dict(image_id=row["id"], camera=row["camera"], part=name, truth=obj["status"],
                                    gt=obj["point_xy"], pred=xy[j].tolist(), confidence=float(conf[j]),
                                    error_px=math.dist(xy[j], obj["point_xy"]) if obj["status"] == "visible" else None))
        for mode in (["heatmap", "spatial_with_R3_visibility"] if old_scores else ["heatmap"]):
            candidate = copy.deepcopy(records)
            if mode != "heatmap":
                for r in candidate:
                    r["confidence"] = old_scores[r["image_id"]][r["part"]]
            tag = str(checkpoint) + ":" + mode
            all_metrics[tag] = {}
            for cam in cameras:
                subset = [r for r in candidate if r["camera"] == cam]
                score = point_metrics(subset, args.error_px)
                all_metrics[tag][cam] = score
                if cam not in choices or score["f1"] > choices[cam]["metric"]["f1"]:
                    choices[cam] = dict(type=mode, source=str(checkpoint), metric=score)
                    picked[cam] = subset
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "weights").mkdir()
    copied = {}
    for cam, spec in choices.items():
        source = Path(spec.pop("source"))
        if source not in copied:
            name = f"spatial_{len(copied):02d}.pt"
            shutil.copy2(source, args.output / "weights" / name)
            copied[source] = "weights/" + name
        spec.update(weights=copied[source], sha256=digest(source))
        if spec["type"] != "heatmap":
            name = cam + "_visibility.pt"
            shutil.copy2(old_paths[cam], args.output / "weights" / name)
            spec.update(visibility_weights="weights/" + name, visibility_sha256=digest(old_paths[cam]))
    save_json(args.output / "camera_routing.json", dict(classes=classes, cameras=choices,
              selection="Per-camera development pixel F1; no independent test claim"))
    save_json(args.output / "inference_contract.json", dict(input=list(input_size), classes=classes))
    save_json(args.output / "candidate_metrics.json", all_metrics)
    save_json(args.output / "validation_predictions.json", sum(picked.values(), []))
    print(args.output)


if __name__ == "__main__":
    main()
