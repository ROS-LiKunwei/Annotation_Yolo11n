"""Single-frame prediction and backward-compatible R4 bundle routing."""
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from .model import HeatmapModel, decode
from .schema import load_json, digest


class Predictor:
    def __init__(self, checkpoint=None, bundle=None, device="cpu"):
        if bool(checkpoint) == bool(bundle):
            raise ValueError("Provide exactly one checkpoint or R4 bundle directory")
        self.device = torch.device(device)
        self.models, self.visibility_models = {}, {}
        self.bundle = Path(bundle).resolve() if bundle else None
        if self.bundle:
            self.routing = load_json(self.bundle / "camera_routing.json")
            self.classes = self.routing["classes"]
            self.input_size = tuple(load_json(self.bundle / "inference_contract.json")["input"])
        else:
            self.checkpoint = Path(checkpoint).resolve()
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.classes = state["classes"]
            self.input_size = tuple(state.get("input", [640, 480]))
            self._load(self.checkpoint, state)

    def _bundle_path(self, name, expected_hash=None):
        path = (self.bundle / name).resolve()
        if not path.is_relative_to(self.bundle):
            raise ValueError("Bundle weight path escapes the bundle")
        if expected_hash and digest(path) != expected_hash:
            raise ValueError(f"Weight SHA256 mismatch: {name}")
        return path

    def _load(self, path, state=None):
        path = Path(path)
        if path not in self.models:
            state = state or torch.load(path, map_location="cpu", weights_only=True)
            if state["classes"] != self.classes:
                raise ValueError("Checkpoint class order mismatch")
            model = HeatmapModel(len(self.classes))
            model.load_state_dict(state["state_dict"], strict=True)
            self.models[path] = model.to(self.device).eval()
        return self.models[path]

    @torch.no_grad()
    def predict_arrays(self, image_bgr, camera=None):
        if image_bgr is None:
            raise ValueError("Cannot read input image")
        height, width = image_bgr.shape[:2]
        spec = None
        if self.bundle:
            if camera not in self.routing["cameras"]:
                raise ValueError("R4 bundle prediction requires a camera from its routing table")
            spec = self.routing["cameras"][camera]
            if spec["type"] not in {"heatmap", "spatial_with_R3_visibility"}:
                raise ValueError(f"Unsupported bundle model type: {spec['type']}")
            if spec["weights"] not in self.models:
                path = self._bundle_path(spec["weights"], spec.get("sha256"))
                model = self._load(path)
                self.models[spec["weights"]] = model
            model = self.models[spec["weights"]]
        else:
            model = self.models[self.checkpoint]
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if (width, height) != self.input_size:
            image = cv2.resize(image, self.input_size)
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(self.device) / 255
        logits, visibility = model(tensor)
        xy, scores = decode(logits, visibility, (width, height))
        xy, scores = xy[0].cpu().numpy(), scores[0].cpu().numpy()
        if spec and spec["type"] == "spatial_with_R3_visibility":
            name = spec["visibility_weights"]
            if name not in self.visibility_models:
                self.visibility_models[name] = YOLO(str(self._bundle_path(name, spec.get("visibility_sha256"))))
            result = self.visibility_models[name].predict(image_bgr, imgsz=640, rect=False, conf=.01,
                                                         max_det=1, device=str(self.device), verbose=False)[0]
            scores = result.keypoints.data[0, :, 2].cpu().numpy() if len(result.boxes) else np.zeros(len(self.classes), np.float32)
            if len(scores) != len(self.classes):
                raise ValueError("Visibility model keypoint order/size mismatch")
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        distribution = logits.float().flatten(2).softmax(-1).view_as(logits)[0].cpu().numpy()
        return xy, scores, distribution

    def predict(self, image_bgr, camera=None, threshold=.5):
        xy, scores, _ = self.predict_arrays(image_bgr, camera)
        return {name: {"point_xy": xy[j].tolist() if scores[j] >= threshold else None,
                       "confidence": float(scores[j]), "pred_visible": bool(scores[j] >= threshold)}
                for j, name in enumerate(self.classes)}
