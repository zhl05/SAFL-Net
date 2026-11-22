"""Evaluate an SAFL-Net checkpoint using reference point and box prompts."""
import argparse
from pathlib import Path

import torch

from safl.backend import build_model, ModelBackendUnavailable
from safl.data import UltrasoundDataset
from safl.engine import evaluate, make_loader, seed_everything
from safl.io import write_json, write_rows
from safl.losses import ClassificationLoss, SegmentationLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--save-masks", action="store_true", help="Also compute boundary distances in resized pixels")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new or empty output directory")
    seed_everything(1234)
    try:
        model = build_model()
    except ModelBackendUnavailable as error:
        parser.exit(2, f"Unavailable: {error}\n")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    model = model.to(device)
    dataset = UltrasoundDataset(args.data_root, args.manifest)
    loader = make_loader(dataset, args.batch_size, args.workers)
    metrics, records = evaluate(model, loader, SegmentationLoss().to(device),
                                ClassificationLoss().to(device), device,
                                args.output / "masks" if args.save_masks else None)
    write_json(args.output / "metrics.json", metrics)
    write_rows(args.output / "predictions.csv", records)
    print(metrics)


if __name__ == "__main__":
    main()
