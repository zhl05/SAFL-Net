"""Binary segmentation/classification metrics and explicit empty-mask handling."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def boundary_distances(prediction, truth):

    if not prediction.any() and not truth.any():

        return 0.0, 0.0, False

    if not prediction.any() or not truth.any():

        return None, None, True

    pred_border = prediction ^ binary_erosion(prediction)

    true_border = truth ^ binary_erosion(truth)

    distances = np.concatenate([distance_transform_edt(~true_border)[pred_border],

                                distance_transform_edt(~pred_border)[true_border]])

    return float(np.percentile(distances, 95)), float(distances.mean()), False


def summarize(records, validation_loss=None):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation")
    labels = [row["label"] for row in records]
    scores = [row["score"] for row in records]
    predicted = [int(value >= 0.5) for value in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    result = dict(dice=float(np.mean([r["dice"] for r in records])),
                  iou=float(np.mean([r["iou"] for r in records])),
                  auc=float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
                  accuracy=float(accuracy_score(labels, predicted)),
                  f1=float(f1_score(labels, predicted, zero_division=0)),
                  precision=float(tp / max(1, tp + fp)),
                  sensitivity=float(tp / max(1, tp + fn)),
                  specificity=float(tn / max(1, tn + fp)), n=len(records))
    if validation_loss is not None:
        result["validation_loss"] = float(validation_loss)
    if "hd95_px" in records[0]:
        valid = [row for row in records if row["hd95_px"] is not None]
        result.update(hd95_px=float(np.mean([r["hd95_px"] for r in valid])) if valid else None,
                      assd_px=float(np.mean([r["assd_px"] for r in valid])) if valid else None,
                      empty_failures=sum(r["empty_failure"] for r in records),
                      distance_policy="Concatenated surface distances at resized image resolution; "
                                      "one-empty excluded and counted; both-empty=0; units=pixels")
    return result
