import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import LateFusionResNet18, CrossAttnFusionResNet18
from dataset import RGBDepthClassificationDataset
from corruptions import depth_dropout, rgb_occlusion, rgb_dark, rgb_blur


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for rgb, depth, y in loader:
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
        logits = model(rgb, depth)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def evaluate_robustness(model, loader, device):
    """
    Two simple stress tests:
      1) depth dropout (zero)
      2) rgb occlusion + darken + blur (choose one or run all)
    """
    model.eval()

    def _acc_with(transform_rgb=None, transform_depth=None):
        correct = 0
        total = 0
        for rgb, depth, y in loader:
            rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)
            if transform_rgb is not None:
                rgb = transform_rgb(rgb)
            if transform_depth is not None:
                depth = transform_depth(depth)
            logits = model(rgb, depth)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)

    acc_depth_zero = _acc_with(transform_depth=lambda d: depth_dropout(d, "zero"))
    acc_rgb_occ = _acc_with(transform_rgb=lambda x: rgb_occlusion(x, frac=0.25))
    acc_rgb_dark = _acc_with(transform_rgb=lambda x: rgb_dark(x, factor=0.3))
    acc_rgb_blur = _acc_with(transform_rgb=lambda x: rgb_blur(x, k=7))

    return {
        "depth_zero": acc_depth_zero,
        "rgb_occlusion": acc_rgb_occ,
        "rgb_dark": acc_rgb_dark,
        "rgb_blur": acc_rgb_blur,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["late", "cross"], default="late")
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = RGBDepthClassificationDataset(split="train")
    val_ds = RGBDepthClassificationDataset(split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    if args.model == "late":
        model = LateFusionResNet18(num_classes=args.num_classes, pretrained=args.pretrained, depth_in_ch=1)
    else:
        model = CrossAttnFusionResNet18(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            depth_in_ch=1,
            num_heads=8,
            bidirectional=args.bidirectional,
        )

    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{args.epochs}")
        for rgb, depth, y in pbar:
            rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(rgb, depth)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

        acc = evaluate(model, val_loader, device)
        print(f"[val] epoch={ep} acc={acc:.4f}")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "args": vars(args)}, "best.pt")
            print(f"  saved best.pt (acc={best:.4f})")

    # Robustness on val set
    rb = evaluate_robustness(model, val_loader, device)
    print("[robustness]", rb)


if __name__ == "__main__":
    main()