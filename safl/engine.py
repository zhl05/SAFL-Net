"""Shared training and evaluation routines."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import seed_worker
from .losses import joint_loss
from .metrics import boundary_distances, summarize


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, batch_size=8, workers=0, seed=1234, training=False):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training,
                        num_workers=workers, pin_memory=torch.cuda.is_available(),
                        drop_last=training, worker_init_fn=seed_worker, generator=generator)
    if len(loader) == 0:
        raise ValueError("No batches: reduce batch size or provide more training cases")
    return loader


def inputs(batch, device):
    image = batch["image"].to(device, dtype=torch.float32)
    masks = batch["low_mask"].to(device, dtype=torch.float32)
    labels = batch["cls_label"].to(device, dtype=torch.long)
    boxes = torch.as_tensor(batch["bbox"], device=device, dtype=torch.float32)
    points = (torch.as_tensor(batch["pt"], device=device, dtype=torch.float32),
              torch.as_tensor(batch["p_label"], device=device, dtype=torch.int))
    return image, masks, labels, boxes, points


def train_one_epoch(model, loader, optimizer, scheduler, scaler, segmentation, classification, device):
    model.train()
    running = np.zeros(3)
    for batch in loader:
        image, masks, labels, boxes, points = inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output, logits, _, _ = model(image, points, boxes, use_auto_prompt=False)
        loss, seg, cls = joint_loss(output, logits, masks, labels, segmentation, classification)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scaler.get_scale() >= old_scale:
            scheduler.step()
        running += [loss.item(), seg.item(), cls.item()]
    return dict(zip(("train_loss", "train_seg_loss", "train_cls_loss"), (running / len(loader)).tolist()))


@torch.no_grad()
def evaluate(model, loader, segmentation, classification, device, mask_dir=None):
    model.eval()
    records, total_loss = [], 0.0
    for batch in loader:
        image, masks, labels, boxes, points = inputs(batch, device)
        # Evaluation is FP32 and requires GT-derived point and box prompts.
        output, logits, _, _ = model(image, points, boxes, use_auto_prompt=False)
        loss, _, _ = joint_loss(output, logits, masks, labels, segmentation, classification)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        total_loss += loss.item() * len(image)
        predicted = (torch.sigmoid(output["masks"])[:, 0] > 0.5).cpu().numpy()
        truth = batch["label"][:, 0].cpu().numpy() > 0.5
        scores = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        if predicted.shape != truth.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid full-resolution predictions or class probabilities")
        for index in range(len(image)):
            prediction, target = predicted[index], truth[index]
            intersection = np.logical_and(prediction, target).sum()
            union = np.logical_or(prediction, target).sum()
            mass = prediction.sum() + target.sum()
            row = dict(image_name=batch["image_name"][index], label=int(labels[index]),
                       score=float(scores[index]), dice=float((2 * intersection + 1e-5) / (mass + 1e-5)),
                       iou=float((intersection + 1e-5) / (union + 1e-5)))
            if mask_dir is not None:
                path = Path(mask_dir) / row["image_name"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), prediction.astype(np.uint8) * 255):
                    raise OSError("Failed to write prediction PNG")
                hd95, assd, failure = boundary_distances(prediction, target)
                row.update(hd95_px=hd95, assd_px=assd, empty_failure=failure)
            records.append(row)
    return summarize(records, total_loss / len(records)), records
