"""Export frozen 64-D encoder features and classifier logits from a trained checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitfd import SourceClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen features from a SourceClassifier checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--input", required=True, help="Input .npz containing x [N, C, L].")
    parser.add_argument("--out", required=True, help="Output feature .npz path.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SourceClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device).eval()

    with np.load(args.input, allow_pickle=False) as f:
        x = np.asarray(f["x"], dtype=np.float32)
        meta = {k: f[k].copy() for k in f.files if k != "x"}
    if "normalization_mean" in checkpoint and "normalization_scale" in checkpoint:
        x = ((x - checkpoint["normalization_mean"]) / checkpoint["normalization_scale"]).astype(np.float32)
    if x.ndim != 3 or len(x) == 0 or not np.isfinite(x).all():
        parser.error("x must be a nonempty finite [N, C, L] array.")

    zs, logits = [], []
    with torch.inference_mode():
        for i in range(0, len(x), args.batch_size):
            z = model.encoder(torch.from_numpy(x[i : i + args.batch_size]).to(args.device))
            zs.append(z.cpu().numpy())
            logits.append(model.head(z).cpu().numpy())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **meta, z=np.concatenate(zs), logits=np.concatenate(logits))
    print(f"Saved {len(x)} frozen feature vectors to {args.out}")


if __name__ == "__main__":
    main()
