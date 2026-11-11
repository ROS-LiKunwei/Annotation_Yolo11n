import copy
from argparse import Namespace
from pathlib import Path
import json
import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from annotation_yolo11n.schema import validate, digest
from annotation_yolo11n.kit import prepare, render
from annotation_yolo11n.losses import point_loss
from annotation_yolo11n.model import HeatmapModel, decode
from annotation_yolo11n.metrics import point_metrics


torch.set_num_threads(2)


def make_dataset(root):
    rows = []
    for ep, split in [(0, "train"), (1, "train"), (2, "val")]:
        path = root / f"image{ep}.jpg"
        Image.new("RGB", (64, 64), (40 + ep * 30, 80, 90)).save(path)
        rows.append(dict(id=f"ep{ep}", file_name=path.name, episode_index=ep, frame_index=0,
                         camera="cam_head", width=64, height=64, split=split, reviewed=True,
                         rgb_sha256=digest(path), objects={
                             "coin": dict(status="visible", point_xy=[20, 30]),
                             "coin_slot": dict(status="not_visible", point_xy=None)}))
    return dict(classes=["coin", "coin_slot"], images=rows, protected_test_episodes=[9])


def test_schema_identity_and_isolation(tmp_path):
    d = make_dataset(tmp_path)
    assert validate(d, tmp_path, for_training=True)["reviewed"] == 3
    bad = copy.deepcopy(d)
    bad["images"][2]["episode_index"] = 0
    with pytest.raises(ValueError, match="crosses"):
        validate(bad)
    bad = copy.deepcopy(d)
    bad["images"][0]["objects"]["coin"]["point_xy"] = [64, 0]
    with pytest.raises(ValueError, match="coordinate"):
        validate(bad)
    bad = copy.deepcopy(d)
    bad["training_allowed"] = False
    with pytest.raises(ValueError, match="review only|review"):
        validate(bad, for_training=True)
    bad = copy.deepcopy(d)
    bad["images"][0]["rgb_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        validate(bad, tmp_path)


def test_unknowns_mask_both_losses_and_absence_only_trains_visibility():
    torch.manual_seed(3)
    hm = torch.randn(2, 3, 12, 16, requires_grad=True)
    vis = torch.zeros(2, 3, requires_grad=True)
    points = torch.full((2, 3, 2), 5.)
    status = torch.tensor([[2, 0, 1], [2, 0, 1]])
    loss = point_loss(hm, vis, points, status)
    loss.backward()
    assert hm.grad[:, 0].abs().sum() > 0
    assert hm.grad[:, 1:].abs().sum() == 0
    assert vis.grad[:, 2].abs().sum() == 0
    assert (vis.grad[:, 0] < 0).all() and (vis.grad[:, 1] > 0).all()
    changed = hm.detach().clone()
    changed[:, 2] += torch.randn_like(changed[:, 2]) * 100
    changed_vis = vis.detach().clone()
    changed_vis[:, 2] = 100
    assert torch.allclose(loss.detach(), point_loss(changed, changed_vis, points, status))


def test_decode_and_arbitrary_class_count():
    hm = torch.zeros(1, 3, 120, 160)
    hm[:, :, 30, 40] = 30
    xy, score = decode(hm, torch.zeros(1, 3))
    assert torch.allclose(xy[0, 0], torch.tensor([160., 120.]), atol=.01)
    assert torch.all(score == .5)
    model = HeatmapModel(3).eval()
    assert all(p.requires_grad for p in model.layers.parameters())
    with torch.no_grad():
        logits, visibility = model(torch.zeros(1, 3, 64, 96))
    assert logits.shape == (1, 3, 16, 24) and visibility.shape == (1, 3)


def test_metrics_penalize_coordinates_and_exclude_unknown():
    rows = [dict(truth="visible", confidence=.9, error_px=15),
            dict(truth="not_visible", confidence=.9, error_px=None),
            dict(truth="ignore", confidence=.99, error_px=None)]
    assert point_metrics(rows, 10)["correct_points"] == 0
    assert point_metrics(rows, 20)["precision"] == .5
    assert point_metrics(rows)["ignored_points"] == 1


def make_kit(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for ep in range(6):
        path = tmp_path / f"rgb{ep}.jpg"
        Image.new("RGB", (64, 64), (30 + ep * 20, 100, 120)).save(path)
        with h5py.File(source / f"episode_{ep:07d}.hdf5", "w") as f:
            raw = path.read_bytes()
            f.create_dataset("vision/cam_head/colors", data=np.array([raw] * 4, dtype=f"S{len(raw)}"))
            f.create_dataset("additional_info/frequency", data=25)
    config = dict(task="demo", cameras=["cam_head"], classes={
        c: dict(label=c, definition="center </script> literal") for c in ["coin", "coin_slot", "marker"]},
        train_episode_count=2, val_episode_count=1, frames_per_episode=2, seed=3)
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps(config))
    out = tmp_path / "kit"
    prepare(Namespace(config=cfg, source=source, output=out, split_policy=None))
    return out


def test_prepare_and_safe_inline_json(tmp_path):
    out = make_kit(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["images"]) == 6
    assert validate(manifest, out)["reviewed"] == 0
    html = render(manifest)
    text = html.split('<script id="manifest" type="application/json">')[1].split('</script>')[0]
    assert json.loads(text) == manifest
    assert "fold_clothes" not in html


def test_one_epoch_cpu_train_and_checkpoint_roundtrip(tmp_path):
    from annotation_yolo11n.train import train
    from annotation_yolo11n.inference import Predictor
    import cv2
    dataset = make_dataset(tmp_path)
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps(dataset))
    args = Namespace(threads=2, seed=42, annotations=source, images_root=tmp_path, output=tmp_path/"run",
                     device="cpu", backbone="yolo11n-pose.yaml", initial_checkpoint=None,
                     input_size=[64, 64], epochs=1, eval_every=1, batch_size=2,
                     backbone_lr=.00005, head_lr=.0003, error_px=10, no_augment=True)
    train(args)
    assert (args.output / "completed.json").exists()
    predictor = Predictor(checkpoint=args.output/"best.pt", device="cpu")
    output = predictor.predict(cv2.imread(str(tmp_path/"image2.jpg")))
    assert set(output) == {"coin", "coin_slot"}


def test_same_rgb_across_splits_is_rejected(tmp_path):
    d = make_dataset(tmp_path)
    d["images"][2]["rgb_sha256"] = d["images"][0]["rgb_sha256"]
    with pytest.raises(ValueError, match="RGB hashes"):
        validate(d, for_training=True)


def test_candidate_and_route_cli_roundtrip(tmp_path):
    import subprocess
    import sys
    from annotation_yolo11n.inference import Predictor
    from annotation_yolo11n.kit import propose
    root = Path(__file__).resolve().parents[1]
    kit = make_kit(tmp_path)
    data = json.loads((kit/"manifest.json").read_text())
    for row in data["images"]:
        row["reviewed"] = True
        row["objects"] = {c: dict(status="visible", point_xy=[30, 30]) for c in data["classes"]}
    labels = tmp_path/"labels.json"
    labels.write_text(json.dumps(data))
    model = HeatmapModel(3)
    checkpoint = tmp_path/"spatial.pt"
    torch.save(dict(classes=data["classes"], input=[64, 64], state_dict=model.state_dict()), checkpoint)
    routed = tmp_path/"routed"
    subprocess.run([sys.executable, str(root/"scripts/select_routing.py"), "--annotations", str(labels),
                    "--images-root", str(kit), "--checkpoints", str(checkpoint), "--output", str(routed)],
                   cwd=root, check=True, capture_output=True, text=True)
    predictor = Predictor(bundle=routed)
    assert predictor.classes == data["classes"]
    # Draft human coordinates survive; only pending labels get model candidates.
    row = data["images"][0]
    row["reviewed"] = False
    row["objects"]["coin_slot"] = dict(status="pending", point_xy=None)
    row["objects"]["marker"] = dict(status="ignore", point_xy=None)
    pending = tmp_path/"pending.json"
    pending.write_text(json.dumps(data))
    output = tmp_path/"candidates"
    propose(Namespace(annotations=pending, images_root=kit, checkpoint=checkpoint, bundle=None,
                      output=output, device="cpu", threshold=1.))
    result = json.loads((output/"manifest.json").read_text())
    first = result["images"][0]
    assert first["objects"]["coin"]["point_xy"] == [30, 30]
    assert first["objects"]["marker"]["status"] == "ignore"
    assert first["objects"]["coin_slot"]["status"] == "pending"
    assert first["reviewed"] is False
    assert result["kit_id"] != data["kit_id"]
