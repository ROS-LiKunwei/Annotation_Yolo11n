"""Command line entrypoints; no workstation paths or task names are hardcoded."""
import argparse
from pathlib import Path
from .schema import load_json, save_json, validate


def prediction(args):
    import cv2
    import numpy as np
    import torch
    from .inference import Predictor
    torch.set_num_threads(args.threads)
    model = Predictor(args.checkpoint, args.bundle, args.device)
    xy, scores, maps = model.predict_arrays(cv2.imread(str(args.image)), args.camera)
    result = {c: {"point_xy": xy[j].tolist() if scores[j] >= args.threshold else None,
                  "confidence": float(scores[j]), "pred_visible": bool(scores[j] >= args.threshold)}
              for j, c in enumerate(model.classes)}
    save_json(args.output, result)
    if args.heatmaps:
        Path(args.heatmaps).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.heatmaps, spatial_distribution=maps, visibility_confidence=scores,
                            valid_mask=scores >= args.threshold, classes=np.array(model.classes))
    print(result)


def model_options(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--bundle", type=Path, help="Original portable R4 bundle with routing and weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=4)


def main():
    parser = argparse.ArgumentParser(description="YOLO11n semantic point annotation and spatial heatmaps")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Extract RoboDojo RGB and create an offline annotation kit")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-policy", type=Path)
    p = sub.add_parser("validate", help="Check coordinate/status/hash/split contracts")
    p.add_argument("annotations", type=Path)
    p.add_argument("--images-root", type=Path)
    p.add_argument("--for-training", action="store_true")
    p = sub.add_parser("train", help="Train the R4 point head; distinct from PaliGemma/VLM20k")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backbone", default="yolo11n-pose.pt", help="YOLO11n pose feature initialization")
    p.add_argument("--initial-checkpoint", type=Path, help="Optional spatial weights; starts a fresh optimizer")
    p.add_argument("--input-size", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--backbone-lr", type=float, default=.00005)
    p.add_argument("--head-lr", type=float, default=.0003)
    p.add_argument("--error-px", type=float, default=10.)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-augment", action="store_true", help="Disable mirror/affine/color; use for direction-sensitive tasks")
    p = sub.add_parser("predict", help="Predict original-pixel points and optional spatial maps")
    p.add_argument("image", type=Path)
    model_options(p)
    p.add_argument("--camera")
    p.add_argument("--output", type=Path, default=Path("prediction.json"))
    p.add_argument("--heatmaps", type=Path)
    p = sub.add_parser("propose", help="Create an unreviewed candidate kit from a pending export")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    model_options(p)
    args = parser.parse_args()
    if hasattr(args, "threshold") and not 0 <= args.threshold <= 1:
        parser.error("threshold must be in[0,1]")
    if args.command == "validate":
        print(validate(load_json(args.annotations), args.images_root, args.for_training))
    elif args.command == "train":
        if min(args.epochs, args.eval_every, args.batch_size, args.threads) <= 0:
            parser.error("epochs, eval-every, batch-size and threads must be positive")
        from .train import train
        train(args)
    elif args.command == "predict":
        prediction(args)
    else:
        from .kit import prepare, propose
        if args.command == "propose":
            import torch
            torch.set_num_threads(args.threads)
        (prepare if args.command == "prepare" else propose)(args)


if __name__ == "__main__":
    main()
