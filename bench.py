import argparse
import time
import torch

from models import LateFusionResNet18, CrossAttnFusionResNet18


@torch.no_grad()
def benchmark(model, device, batch_size: int, iters: int = 200, warmup: int = 30, h: int = 224, w: int = 224):
    model.eval()
    rgb = torch.rand(batch_size, 3, h, w, device=device)
    depth = torch.rand(batch_size, 1, h, w, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(rgb, depth)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(rgb, depth)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    latency_ms = (total / iters) * 1000.0
    throughput = (batch_size * iters) / total

    peak_gb = None
    if device == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3)

    return latency_ms, throughput, peak_gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["late", "cross"], default="late")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--bidirectional", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    for bs in [1, 8]:
        lat, thr, mem = benchmark(model, device, batch_size=bs)
        print(f"bs={bs} latency_ms={lat:.2f} throughput={thr:.1f} samples/s peak_vram_gb={mem}")


if __name__ == "__main__":
    main()