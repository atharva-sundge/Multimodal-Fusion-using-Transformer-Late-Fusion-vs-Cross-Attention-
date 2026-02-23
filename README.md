# Multimodal-Fusion-using-Transformer-Late-Fusion-vs-Cross-Attention-
How to fuse multiple modalities (camera, LiDAR, radar) so the model can reason better especially under real-world conditions and compute resources constriants

Details:
Implemented token fusion via cross-attention on spatial feature-map tokens (ResNet18 → tokens).

Compared against a strong late fusion baseline with identical encoders.

Reported accuracy + latency + throughput + peak VRAM on L4/L40 with correct GPU benchmarking.

Tested robustness with depth dropout and RGB corruptions (occlusion/dark/blur).

Summarized when cross-attention is worth the compute tradeoff.
