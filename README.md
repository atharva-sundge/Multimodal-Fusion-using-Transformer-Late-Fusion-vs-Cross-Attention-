# Multimodal-Fusion-using-Transformer-Late-Fusion-vs-Cross-Attention-
How to fuse multiple modalities (camera, LiDAR, radar) so the model can reason better especially under real-world conditions and compute resources constriants

Details:
1]
Implemented token fusion via cross-attention on spatial feature-map tokens (ResNet18 → tokens).

2] Compared against a strong late fusion baseline with identical encoders.

3] Reported accuracy + latency + throughput + peak VRAM on L4/L40 with correct GPU benchmarking.

4] Tested robustness with depth dropout and RGB corruptions (occlusion/dark/blur).

5] Summarized when cross-attention is worth the compute tradeoff.
