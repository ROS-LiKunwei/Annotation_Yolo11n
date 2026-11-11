"""Portable R4 spatial training. Only reviewed train rows enter optimizer batches."""
import math
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import PointDataset
from .model import HeatmapModel, decode
from .losses import point_loss
from .metrics import summarize
from .schema import load_json, save_json, validate, digest


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=8):
    model.eval()
    records, cursor = [], 0
    for images, _, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits, visibility = model(images.to(device))
        for j in range(len(images)):
            row = dataset.rows[cursor + j]
            xy, conf = decode(logits[j:j+1], visibility[j:j+1], (row["width"], row["height"]))
            for name, point, score in zip(dataset.classes, xy[0].cpu().tolist(), conf[0].cpu().tolist()):
                obj = row["objects"][name]
                records.append(dict(image_id=row["id"], episode_index=row["episode_index"], camera=row["camera"],
                                    part=name, truth=obj["status"], gt=obj["point_xy"], pred=point, confidence=score,
                                    error_px=math.dist(point, obj["point_xy"]) if obj["status"] == "visible" else None))
        cursor += len(images)
    return records


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = load_json(args.annotations)
    audit = validate(data, args.images_root, for_training=True)
    reviewed = [r for r in data["images"] if r["reviewed"]]
    train_rows = [r for r in reviewed if r["split"] == "train"
                  and any(o["status"] in {"visible", "not_visible"} for o in r["objects"].values())]
    val_rows = [r for r in reviewed if r["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Reviewed train and validation images are both required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "intake_audit.json", {**audit, "train_images": len(train_rows), "val_images": len(val_rows),
              "unreviewed_excluded": len(data["images"]) - len(reviewed), "annotation_sha256": digest(args.annotations)})
    shutil.copy2(args.annotations, out / "human_labels.json")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    config["classes"] = data["classes"]
    save_json(out / "config.json", config)
    device = torch.device(args.device)
    model = HeatmapModel(len(data["classes"]), args.backbone).to(device)
    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["classes"] != data["classes"]:
            raise ValueError("Initial checkpoint class order mismatch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    heads = [p for name, p in model.named_parameters() if not name.startswith("layers.")]
    optimizer = torch.optim.AdamW([{"params": model.layers.parameters(), "lr": args.backbone_lr},
                                  {"params": heads, "lr": args.head_lr}], weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.000005)
    size = tuple(args.input_size)
    train_set = PointDataset(train_rows, data["classes"], args.images_root, size, augment=not args.no_augment)
    val_set = PointDataset(val_rows, data["classes"], args.images_root, size)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    history, best = [], -1.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = []
        for images, points, status in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, visibility = model(images.to(device))
                loss = point_loss(logits, visibility, points.to(device), status.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if not any(p.grad is not None for p in model.layers.parameters()):
                raise RuntimeError("Pretrained backbone unexpectedly frozen")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            total.append(float(loss.detach()))
        scheduler.step()
        history.append({"epoch": epoch, "loss": float(np.mean(total))})
        save_json(out / "progress.json", {**history[-1], "max_epochs": args.epochs, "status": "training"})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            records = evaluate(model, val_set, device, args.batch_size)
            report = summarize(records, args.error_px)
            save_json(out / f"validation_epoch{epoch}.json", report)
            save_json(out / f"predictions_epoch{epoch}.json", records)
            checkpoint = dict(schema="annotation_yolo11n_spatial_v1", model_type="yolo11n_spatial_heatmap_v1",
                              state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              classes=data["classes"], input=list(size), epoch=epoch, config=config)
            torch.save(checkpoint, out / f"epoch{epoch}.pt")
            if report["overall"]["f1"] > best:
                best = report["overall"]["f1"]
                torch.save(checkpoint, out / "best.pt")
                save_json(out / "best_evaluation.json", report)
            print(f"epoch={epoch} loss={np.mean(total):.4f} F1={report['overall']['f1']:.4f}", flush=True)
    save_json(out / "history.json", history)
    save_json(out / "completed.json", {"epochs": args.epochs, "best_f1": best,
              "selection": "Same development validation used for checkpoint selection; not independent testing"})
