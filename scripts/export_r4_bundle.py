#!/usr/bin/env python3
"""Copy only inference files from an existing private R4 bundle. Weights stay external to Git."""
import argparse
import shutil
from pathlib import Path
from annotation_yolo11n.schema import load_json, digest

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
route = load_json(a.source / "camera_routing.json")
paths = {"camera_routing.json", "inference_contract.json"}
for spec in route["cameras"].values():
    for field, hashfield in [("weights", "sha256"), ("visibility_weights", "visibility_sha256")]:
        if field in spec:
            file = (a.source / spec[field]).resolve()
            if not file.is_relative_to(a.source.resolve()):
                raise ValueError("Source weight must be inside bundle")
            if spec.get(hashfield) and digest(file) != spec[hashfield]:
                raise ValueError("Weight hash mismatch")
            paths.add(spec[field])
a.output.mkdir(parents=True, exist_ok=False)
for name in paths:
    dst = a.output / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.source / name, dst)
print(a.output)
