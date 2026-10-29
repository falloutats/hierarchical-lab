"""Stage 5 — watch the representation organize itself by superclass.

Train the coarse-to-fine model with layer loss + the differentiable soft
hierarchy loss, and after every epoch snapshot the penultimate features of a
fixed test subset. The soft loss (not the head-agreement loss from Stage 4) is
the right choice here: we want to watch features group by their TRUE superclass,
and that is exactly what it rewards. Render a 2-D PCA GIF colored by superclass plus a centroid-
trajectory PNG — the prototypical-toy "watch it evolve" experience, on a real
20-way taxonomy.

    uv run python s5_representation_evolution.py --epochs 12

Writes runs/embedding_evolution.gif and runs/embedding_trajectory.png
"""

from __future__ import annotations

import argparse
import itertools

import torch
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from losses import SoftHierarchyLoss, accuracy, layer_loss, violation_rate
from models import CoarseToFineModel
from viz import EmbeddingEvolution


@torch.no_grad()
def snapshot_features(model, subset, device):
    model.eval()
    feats, coarse = [], []
    for x, c, _f in subset:
        feats.append(model.features(x.to(device)).cpu())
        coarse.append(c)
    return torch.cat(feats).numpy(), torch.cat(coarse).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    train_loader, test_loader = make_loaders(batch_size=args.batch_size)
    # Fixed subset of the test set for consistent snapshots across epochs. The
    # test loader is unshuffled, so these are the same 2048 images every time;
    # islice stops after 8 batches instead of loading the whole test set.
    subset = list(itertools.islice(test_loader, 8))

    model = CoarseToFineModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    soft = SoftHierarchyLoss(beta=1.0).to(device)
    eco = EmbeddingEvolution()

    # Snapshot the random-init representation (epoch 0) so the GIF starts from chaos.
    f0, c0 = snapshot_features(model, subset, device)
    eco.record(0, f0, c0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coarse, fine in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
            opt.zero_grad()
            c_logits, f_logits = model(x)
            loss = layer_loss(c_logits, f_logits, coarse, fine) + soft(f_logits, coarse)
            loss.backward()
            opt.step()

        feats, coarse_np = snapshot_features(model, subset, device)
        eco.record(epoch, feats, coarse_np)

        # cheap per-epoch scalar readout
        model.eval()
        cl, fl, ct, ft = [], [], [], []
        with torch.no_grad():
            for x, c, f in test_loader:
                a, b = model(x.to(device))
                cl.append(a.cpu())
                fl.append(b.cpu())
                ct.append(c)
                ft.append(f)
        cl, fl, ct, ft = torch.cat(cl), torch.cat(fl), torch.cat(ct), torch.cat(ft)
        print(f"epoch {epoch}: fine acc {accuracy(fl, ft):.2f}%  "
              f"coarse acc {accuracy(cl, ct):.2f}%  viol {violation_rate(cl, fl):.2f}%")

    gif = eco.save_gif()
    traj = eco.save_trajectory()
    print(f"wrote {gif}\nwrote {traj}")


if __name__ == "__main__":
    main()
