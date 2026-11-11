"""Pixel precision/recall includes localization errors and ignores unknown labels."""
import numpy as np


def point_metrics(records, distance=10, threshold=.5):
    known = [r for r in records if r["truth"] in {"visible", "not_visible"}]
    visible = [r for r in known if r["truth"] == "visible"]
    proposed = [r for r in known if r["confidence"] >= threshold]
    correct = [r for r in proposed if r["truth"] == "visible" and r["error_px"] <= distance]
    errors = [r["error_px"] for r in visible if r["confidence"] >= threshold]
    return dict(known_points=len(known), visible_points=len(visible), proposed_points=len(proposed),
                correct_points=len(correct), precision=len(correct) / len(proposed) if proposed else None,
                recall=len(correct) / len(visible) if visible else None,
                f1=2 * len(correct) / (len(visible) + len(proposed)) if visible or proposed else 0.,
                invisible_false_positives=sum(r["truth"] == "not_visible" for r in proposed),
                visible_missed=sum(r["confidence"] < threshold for r in visible),
                median_error_px=float(np.median(errors)) if errors else None,
                mean_error_px=float(np.mean(errors)) if errors else None,
                ignored_points=len(records) - len(known), distance_px=distance, confidence_threshold=threshold)


def summarize(records, distance=10):
    return dict(overall=point_metrics(records, distance),
                by_camera={c: point_metrics([r for r in records if r["camera"] == c], distance)
                           for c in sorted({r["camera"] for r in records})},
                by_class={c: point_metrics([r for r in records if r["part"] == c], distance)
                          for c in sorted({r["part"] for r in records})})
