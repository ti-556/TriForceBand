import argparse, json, os, random

import numpy as np, h5py, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import wandb

from model.triforcemodel import (
    MultiFileXelaDataset,
    SpatialMixerXS,
    compute_normalization_stats,
    create_balanced_sampler,
    split_indices,
    mixup_collate,
    run_epoch,
)

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # WandB
    wandb.init(project=args.wandb_project,
               name=args.wandb_name,
               config=vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    h5_paths = args.h5
    with h5py.File(h5_paths[0], "r") as hf0:
        label_map = json.loads(hf0.attrs["label_mapping"])
    inv_map = {v: k for k, v in label_map.items()}

    hfs     = [h5py.File(p, "r") for p in h5_paths]
    lengths = [hf["y"].shape[0] for hf in hfs]
    T_vals  = {hf["X"].shape[-1] for hf in hfs}
    if len(T_vals) != 1:
        raise ValueError("All HDF5 files must have identical time dimension T")
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    y_all   = np.concatenate([hf["y"][:] for hf in hfs])
    for hf in hfs: hf.close()

    n_classes = len(inv_map)
    uniq, cnts = np.unique(y_all, return_counts=True)
    print("Class distribution:")
    for cid, c in zip(uniq, cnts):
        print(f"  {inv_map[cid]} ({cid}): {c} samples ({c/len(y_all)*100:.1f}%)")

    tr_idx, va_idx, te_idx = split_indices(y_all, seed=args.seed)
    print("Computing normalization stats...")
    mean, std = compute_normalization_stats(
        h5_paths, offsets, tr_idx, args.normal_only
    )
    np.savez(args.stats, mean=mean, std=std)
    print(f"✔ Stats saved to {args.stats}")

    train_ds = MultiFileXelaDataset(
        h5_paths, offsets, tr_idx, y_all,
        train=True,  normalize_stats=(mean, std),
        normal_only=args.normal_only
    )
    val_ds   = MultiFileXelaDataset(
        h5_paths, offsets, va_idx, y_all,
        train=False, normalize_stats=(mean, std),
        normal_only=args.normal_only
    )
    test_ds  = MultiFileXelaDataset(
        h5_paths, offsets, te_idx, y_all,
        train=False, normalize_stats=(mean, std),
        normal_only=args.normal_only
    )

    collate_fn = (lambda b: mixup_collate(b, 0.4)) if args.mixup else None
    if args.balanced_sampling:
        sampler = create_balanced_sampler(y_all[tr_idx])
        train_dl = DataLoader(train_ds, batch_size=args.bs,
                              sampler=sampler, num_workers=4,
                              collate_fn=collate_fn)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.bs,
                              shuffle=True, num_workers=4,
                              collate_fn=collate_fn)
    val_dl  = DataLoader(val_ds,  batch_size=args.bs, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    in_ch = 1 if args.normal_only else 3
    model = SpatialMixerXS(n_classes, in_ch=in_ch, dropout=args.dropout).to(device)
    wandb.config.update({"input_channels": in_ch})
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    wandb.log({"model/total_parameters": total_params})

    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=8
    )

    ce_weight = None
    if args.use_class_weights:
        uniq, cnts = np.unique(y_all[tr_idx], return_counts=True)
        ce_weight = torch.tensor(len(tr_idx) / (len(uniq) * cnts),
                                 dtype=torch.float32).to(device)
        print("CE class weights:", ce_weight.cpu().numpy())

    loss_fn = nn.CrossEntropyLoss(weight=ce_weight)

    best_val, patience = 1e9, 20
    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_dl, opt, device, loss_fn, args.mixup
        )
        va_loss, va_acc = run_epoch(
            model, val_dl, None, device, loss_fn
        )

        wandb.log({
            "epoch": ep,
            "train/loss": tr_loss, "train/accuracy": tr_acc,
            "val/loss": va_loss,   "val/accuracy": va_acc,
            "lr": opt.param_groups[0]["lr"]
        })
        scheduler.step(va_loss)

        improved = va_loss < best_val
        best_val = min(best_val, va_loss)
        patience = 20 if improved else patience - 1
        print(f"[{ep:03d}] tr_loss {tr_loss:.4f}  va_loss {va_loss:.4f}  "
              f"va_acc {va_acc:.3f}{' ★' if improved else ''}")

        if improved:
            torch.save(model.state_dict(), args.ckpt)
            wandb.log({
                "best/epoch": ep,
                "best/val_loss": va_loss,
                "best/val_accuracy": va_acc
            })
        if patience == 0:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    te_loss, te_acc = run_epoch(model, test_dl, None, device, loss_fn)
    print(f"Test accuracy: {te_acc*100:.2f}%")
    wandb.log({"test/loss": te_loss, "test/accuracy": te_acc})

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_dl:
            out = model(X.to(device)).argmax(1).cpu()
            y_true.append(y)
            y_pred.append(out)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    class_names = [inv_map[i] for i in range(n_classes)]
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    wandb.log({
        "conf_matrix": wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred, class_names=class_names
        )
    })

    # pretty plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(n_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(n_classes), class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Normalized CM")
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, f"{cm[i,j]:.2f}",
                     ha="center", va="center",
                     color="white" if cm[i,j] > 0.5 else "black")
    plt.tight_layout()
    fig_path = os.path.splitext(args.ckpt)[0] + "_confmat.png"
    plt.savefig(fig_path, dpi=180)
    wandb.log({"confmat_img": wandb.Image(fig_path)})

    # upload best model
    art = wandb.Artifact(f"model-{wandb.run.id}", type="model")
    art.add_file(args.ckpt)
    wandb.log_artifact(art)
    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", nargs="+", required=True,
                   help="One or more HDF5 files containing 'X' and 'y'.")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default="best_spatialmixer.pth")
    p.add_argument("--wandb-project", default="xela-gesture-recognition")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--use-class-weights", action="store_true")
    p.add_argument("--balanced-sampling", action="store_true")
    p.add_argument("--normal-only", action="store_true",
                   help="Use only the Z (normal-force) channel.")
    p.add_argument("--mixup", action="store_true")
    p.add_argument("--stats", default="stats.npz",
                   help="Path to save normalization mean/std for inference.")
    args = p.parse_args()
    main(args)
