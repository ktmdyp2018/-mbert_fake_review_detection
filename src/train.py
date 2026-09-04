"""
Training loop (Table 5-B):
    - AdamW for the mBERT backbone (lr=2e-5, weight_decay=0.01)
    - Adam for the FNN head (lr=5e-4)
    - Linear LR decay with warm-up ratio 0.1
    - Gradient clipping at max-norm 1.0
    - Early stopping (patience=3, monitored on validation loss)
    - Domain-adversarial loss (Eq. 9b) ramped in via the GRL schedule
"""

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TRAIN
from src.dataset import ReviewDataset, make_collate_fn
from src.fusion_model import FullPipelineModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(TRAIN.device if torch.cuda.is_available() else "cpu")


def build_optimizer(model: FullPipelineModel):
    mbert_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = (list(model.classifier.parameters()) +
                    list(model.cross_module.parameters()))
    optimizer = torch.optim.AdamW(
        [
            {"params": mbert_params, "lr": TRAIN.mbert_lr, "weight_decay": TRAIN.mbert_weight_decay},
            {"params": head_params, "lr": TRAIN.fnn_lr, "weight_decay": 0.0},
        ]
    )
    return optimizer


def run_epoch(model, loader, optimizer, scheduler, device, train: bool,
              total_steps: int, global_step_offset: int = 0, use_domain_loss: bool = True):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0

    for step, batch in enumerate(loader):
        # BatchNorm1d requires batch size > 1 in training mode; skip a
        # stray final batch of size 1 (can occur with small datasets).
        if train and batch["labels"].size(0) < 2:
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device) if batch["token_type_ids"] is not None else None
        handcrafted = batch["handcrafted"].to(device)
        labels = batch["labels"].to(device)
        domain_labels = batch["domain_labels"].to(device) if use_domain_loss else None

        progress = (global_step_offset + step) / max(total_steps, 1)

        with torch.set_grad_enabled(train):
            logits, C_i = model(input_ids, attention_mask, handcrafted, token_type_ids)
            loss, cls_loss, domain_loss = model.compute_loss(
                logits, labels, C_i=C_i, domain_labels=domain_labels, progress=progress
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(train_df, val_df, seed: int = 42, backbone: str = None,
                 use_domain_loss: bool = True, verbose: bool = True):
    """
    Full training run with early stopping. Returns the best model
    (lowest validation loss) and its training history.
    """
    set_seed(seed)
    device = get_device()

    from src.dataset import compute_class_weights
    class_weights = torch.tensor(
        compute_class_weights(train_df["label"].values), dtype=torch.float32
    ).to(device)

    model = FullPipelineModel(backbone=backbone, class_weights=class_weights).to(device)
    collate = make_collate_fn(model.encoder)

    train_loader = DataLoader(ReviewDataset(train_df), batch_size=TRAIN.batch_size,
                               shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ReviewDataset(val_df), batch_size=TRAIN.batch_size,
                             shuffle=False, collate_fn=collate)

    optimizer = build_optimizer(model)
    total_steps = len(train_loader) * TRAIN.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(TRAIN.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_left = TRAIN.early_stop_patience
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(TRAIN.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device,
                                train=True, total_steps=total_steps,
                                global_step_offset=epoch * len(train_loader),
                                use_domain_loss=use_domain_loss)
        val_loss = run_epoch(model, val_loader, optimizer, None, device,
                              train=False, total_steps=total_steps,
                              use_domain_loss=use_domain_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"[seed={seed}] epoch {epoch+1}/{TRAIN.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = TRAIN.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[seed={seed}] early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
