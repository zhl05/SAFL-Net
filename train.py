"""Train SAFL-Net on one cross-validation fold."""
import argparse
from pathlib import Path

import torch

from safl.backend import build_model, build_optimizer, ModelBackendUnavailable
from safl.data import UltrasoundDataset
from safl.engine import evaluate, make_loader, seed_everything, train_one_epoch
from safl.io import save_checkpoint, write_json, write_rows
from safl.losses import ClassificationLoss, SegmentationLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("Epochs/batch size must be positive and workers nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new or empty output directory")
    seed_everything(args.seed)
    try:
        model = build_model()
    except ModelBackendUnavailable as error:
        parser.exit(2, f"Unavailable: {error}\n")
    device = torch.device(args.device)
    model = model.to(device)
    try:
        optimizer, scheduler = build_optimizer(model)
    except ModelBackendUnavailable as error:
        parser.exit(2, f"Unavailable: {error}\n")
    train_set = UltrasoundDataset(args.data_root, args.train_manifest, training=True)
    val_set = UltrasoundDataset(args.data_root, args.val_manifest)
    train_keys = {(r[1], r[2]) for r in train_set.rows}
    val_keys = {(r[1], r[2]) for r in val_set.rows}
    if train_keys & val_keys:
        parser.error("Training and validation manifests contain overlapping cases")
    validation = make_loader(val_set, args.batch_size, args.workers)
    segmentation = SegmentationLoss().to(device)
    classification = ClassificationLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, enabled=device.type == "cuda")
    best, history = {}, []
    for epoch in range(args.epochs):
        epoch_seed = args.seed + 1009 * epoch
        seed_everything(epoch_seed)
        training = make_loader(train_set, args.batch_size, args.workers, epoch_seed, training=True)
        losses = train_one_epoch(model, training, optimizer, scheduler, scaler,
                                 segmentation, classification, device)
        metrics, records = evaluate(model, validation, segmentation, classification, device)
        record = dict(epoch=epoch + 1, **losses, **metrics)
        history.append(record)
        for selector, secondary in (("dice", "auc"), ("auc", "dice")):
            if metrics[selector] is None:
                continue
            score = (metrics[selector], metrics[secondary] if metrics[secondary] is not None else -1.0)
            if selector not in best or score > best[selector]:
                best[selector] = score
                save_checkpoint(args.output / f"fold{args.fold}_best_{selector}.pth",
                                {"state_dict": model.state_dict(), "fold": args.fold,
                                 "epoch": epoch + 1, "selector": selector, "metrics": metrics})
                write_rows(args.output / f"best_{selector}_predictions.csv", records)
        write_json(args.output / "history.json", history)
        print(record, flush=True)


if __name__ == "__main__":
    main()
