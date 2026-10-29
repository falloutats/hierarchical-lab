"""Stage 4 — the intellectual core.

Part A (the proof): their dependency loss `dloss` is built from argmax + `==` +
where. We run it once and show it has NO gradient path — `requires_grad` is
False, so it cannot even .backward(). It's a monitoring number wearing a loss's
clothes. Then we show SoftHierarchyLoss DOES push gradient into the trunk.

Part B (the experiment): train four identical CoarseToFineModels, same seed and
same data order, and compare the hierarchy-violation curve:

    1. lloss only            baseline, no consistency pressure
    2. lloss + their dloss   dead term; trajectory is IDENTICAL to #1
    3. lloss + soft loss     ties the fine head to the TRUE parent
    4. lloss + agreement KL  ties the two heads to EACH OTHER

Measured result (6 epochs, seed 0 — see the README table): #2 matches #1 to the
last decimal, confirming the dead gradient. #3 does NOT improve the violation
rate, and that negative result is the interesting part: it optimizes fine-vs-
truth consistency while the metric measures fine-vs-coarse-head agreement, so it
can be satisfied while the heads still contradict each other. #4 is the variant
that puts the measured quantity itself into the objective.

The lesson is bigger than the loss: a plausible-sounding regularizer that is
differentiable and correct can still fail, simply because it does not optimize
the thing you are measuring. Check that link before trusting the mechanism.

    uv run python s4_dependency_loss.py --epochs 6

Writes runs/violation_comparison.png
"""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from data import make_loaders
from device import pick_device, set_seed
from losses import (
    HeadAgreementLoss,
    SoftHierarchyLoss,
    accuracy,
    faithful_dloss,
    layer_loss,
    violation_rate,
)
from models import CoarseToFineModel
from viz import plot_violation_comparison


def grad_probe(device) -> None:
    """Part A — demonstrate the dead gradient, then a live one.

    What to watch for. Autograd records a graph only over differentiable ops; the
    node that produced a tensor is stored in `.grad_fn`. argmax and `==` return
    tensors with no grad_fn, because they are piecewise-constant: perturb a logit
    infinitesimally and the output does not move, so the derivative is 0 almost
    everywhere and undefined at the jumps. PyTorch does not raise -- it simply
    hands back a tensor detached from the graph, which is why this bug survives
    code review and a loss curve that looks perfectly reasonable.

    The tell is `requires_grad=False` on a quantity you are about to add to your
    objective. Adding a constant to a loss changes its printed value while leaving
    every gradient identical, so the training run looks fine and learns nothing
    from that term. Part B measures the consequence.
    """
    print("\n=== Part A: does the dependency loss have a gradient? ===")
    model = CoarseToFineModel().to(device)
    train_loader, _ = make_loaders(batch_size=128)
    x, coarse, fine = next(iter(train_loader))
    x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
    c_logits, f_logits = model(x)

    d = faithful_dloss(c_logits, f_logits, coarse, fine)
    print(f"their dloss value        : {d.item():.4f}")
    print(f"their dloss requires_grad: {d.requires_grad}   grad_fn: {d.grad_fn}")
    if not d.requires_grad:
        print("  -> no graph -> .backward() is impossible -> ZERO gradient to every weight.")

    soft = SoftHierarchyLoss().to(device)
    s = soft(f_logits, coarse)
    model.zero_grad()
    s.backward()
    # Global L2 norm over every backbone parameter: sqrt of the summed squares.
    trunk_grad = sum(
        p.grad.pow(2).sum().item()
        for n, p in model.named_parameters()
        if n.startswith("backbone") and p.grad is not None
    ) ** 0.5
    print(f"soft loss  requires_grad : {s.requires_grad}   grad_fn: {type(s.grad_fn).__name__}")
    print(f"soft loss  trunk grad-norm: {trunk_grad:.4f}   -> gradient reaches the backbone.")


def train_variant(name, use_dloss, use_soft, use_agree, args, device):
    set_seed(args.seed)  # identical init/data order per variant for a fair A/B
    train_loader, test_loader = make_loaders(batch_size=args.batch_size)
    model = CoarseToFineModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    soft = SoftHierarchyLoss(beta=1.0).to(device)
    agree = HeadAgreementLoss(gamma=1.0).to(device)

    viol_hist, fine_hist = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coarse, fine in tqdm(train_loader, desc=f"{name} e{epoch}", leave=False):
            x, coarse, fine = x.to(device), coarse.to(device), fine.to(device)
            opt.zero_grad()
            c_logits, f_logits = model(x)
            loss = layer_loss(c_logits, f_logits, coarse, fine)
            if use_dloss:
                loss = loss + faithful_dloss(c_logits, f_logits, coarse, fine)
            if use_soft:
                loss = loss + soft(f_logits, coarse)
            if use_agree:
                loss = loss + agree(c_logits, f_logits)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            cl, fl, ct, ft = [], [], [], []
            for x, coarse, fine in test_loader:
                c_logits, f_logits = model(x.to(device))
                cl.append(c_logits.cpu())
                fl.append(f_logits.cpu())
                ct.append(coarse)
                ft.append(fine)
            cl, fl, ct, ft = torch.cat(cl), torch.cat(fl), torch.cat(ct), torch.cat(ft)
        viol_hist.append(violation_rate(cl, fl))
        fine_hist.append(accuracy(fl, ft))
        print(f"  {name:<16} epoch {epoch}: fine acc {fine_hist[-1]:.2f}%  viol {viol_hist[-1]:.2f}%")
    return viol_hist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    grad_probe(device)

    print("\n=== Part B: violation rate under four losses ===")
    series = {
        "lloss only": train_variant("lloss only", False, False, False, args, device),
        "lloss + their dloss": train_variant("their dloss", True, False, False, args, device),
        "lloss + soft (ties fine to TRUE parent)":
            train_variant("soft loss", False, True, False, args, device),
        "lloss + head agreement KL(m||c)":
            train_variant("agreement", False, False, True, args, device),
    }
    path = plot_violation_comparison(series)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
