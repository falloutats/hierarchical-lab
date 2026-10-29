"""The takeaway toolkit — dataset-agnostic structured-output helpers.

Everything in the pathway that's worth keeping, distilled into one file with no
CIFAR/torchvision dependency. Point it at ANY two-level taxonomy: pass a
`child_to_parent` list (child_to_parent[fine_id] = coarse_id) and you get a
differentiable "respect the tree" loss, a head-agreement loss for two-head
models, and a violation metric.

Where this applies in your future work
---------------------------------------
Any time you know structure among your outputs and want the model to honor it:
  * LLM classification/routing over a label taxonomy (intent -> sub-intent).
  * Cascaded / coarse-to-fine decoders (predict the region, then the point).
  * Product / document categorization, ICD codes, any ontology.
The pattern: (1) share a trunk, (2) supervise every level, (3) enforce
consistency by MARGINALIZING the fine distribution up to the parents and making
that agree with the coarse head's own distribution (HeadAgreementLoss) —
differentiably. Never enforce it with argmax; argmax has no gradient (see
faithful_dloss in losses.py for the trap).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_parent_matrix(child_to_parent: list[int]) -> torch.Tensor:
    """Membership matrix M, shape (num_parents, num_children), M[p, c] = 1 iff c is under p.

    Each column carries exactly one 1, so M is the one-hot encoding of the
    child->parent function. Its usefulness is the identity `probs @ M.T`, which
    sums each parent's children into that parent's slot -- marginalization up the
    tree expressed as a single linear map.
    """
    num_children = len(child_to_parent)
    num_parents = max(child_to_parent) + 1
    m = torch.zeros(num_parents, num_children)
    for child, parent in enumerate(child_to_parent):
        m[parent, child] = 1.0
    return m


class SoftHierarchyLoss(nn.Module):
    """Differentiable "the child must respect its parent" loss.

        loss = -log( probability mass the model placed inside the TRUE parent )

    The design decision worth copying: measure MASS, not the winner. "Is the
    argmax child under the right parent?" is a step function of the logits -- flat
    almost everywhere, so its gradient is zero and it teaches nothing. "How much
    probability landed in the right group?" varies smoothly with every logit, so
    gradient descent can act on it.

    Computed in log-space. The definition is

        -log sum_{c in children(p)} exp(z_c) / sum_j exp(z_j)

    Building probabilities first and then taking a log underflows: once the mass
    in the parent drops below ~1e-45 the sum is exactly 0 in float32 and the log
    is -inf, which is usually patched with an epsilon that quietly caps the loss
    and biases the gradient. log_softmax followed by logsumexp subtracts the max
    internally, so it is exact far into the tail and needs no epsilon -- and the
    tail is exactly where a confidently-wrong model lives.

    Children are selected with masked_fill(-inf) instead of multiplying by M,
    because exp(-inf)=0 drops those terms from the sum cleanly, whereas masking
    probabilities to 0 would place a hard zero inside a logarithm.
    """

    def __init__(self, child_to_parent: list[int], beta: float = 1.0):
        super().__init__()
        self.beta = beta
        # Buffers, not parameters: they must follow .to(device) and land in
        # state_dict, but they are fixed structure and must never get gradients.
        self.register_buffer("M", build_parent_matrix(child_to_parent))
        self.register_buffer("child_to_parent",
                             torch.tensor(child_to_parent, dtype=torch.long))

    def forward(self, fine_logits: torch.Tensor, parent_target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(fine_logits, dim=1)
        # Row p of M indicates p's children, so indexing by the batch of targets
        # yields a (B, num_children) sibling mask in one gather.
        sibling_mask = self.M[parent_target]
        masked = log_p.masked_fill(sibling_mask == 0, float("-inf"))
        return self.beta * (-torch.logsumexp(masked, dim=1)).mean()

    @torch.no_grad()
    def violation_rate(self, fine_logits: torch.Tensor,
                       parent_pred_or_target: torch.Tensor) -> float:
        """% of samples whose argmax child sits under a different parent.

        Pass the coarse head's prediction to measure the model's internal
        self-consistency, or the true parent to measure correctness. Both are
        meaningful; they answer different questions.
        """
        implied_parent = self.child_to_parent[fine_logits.argmax(1)]
        return (implied_parent != parent_pred_or_target).float().mean().item() * 100.0


class HeadAgreementLoss(nn.Module):
    """KL( fine head marginalized to parents || coarse head ), for two-head models.

    SoftHierarchyLoss ties the fine head to the TRUE parent. If what you care about
    is the two heads not contradicting each other, that is the wrong target: put
    the agreement itself in the objective. With m the fine distribution summed
    within each parent and c the coarse head's distribution,

        loss = KL(m || c) = sum_p m_p * (log m_p - log c_p)

    Both heads get gradient: the coarse head is fit to m as a soft target, and the
    fine head is pushed to keep its mass inside parents the coarse head believes
    in. Two heads can agree on a wrong answer, so add this to per-level
    cross-entropy; never use it alone. The group-sum is a logsumexp over masked
    log-probabilities, for the same underflow reasons as SoftHierarchyLoss.
    """

    def __init__(self, child_to_parent: list[int], gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("M", build_parent_matrix(child_to_parent))

    def forward(self, coarse_logits: torch.Tensor, fine_logits: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(fine_logits, dim=1)                          # (B, C)
        # (B, 1, C) against the (1, P, C) mask -> per-parent log-mass, (B, P).
        masked = log_p.unsqueeze(1).masked_fill(self.M.unsqueeze(0) == 0, float("-inf"))
        log_m = torch.logsumexp(masked, dim=2)
        log_c = F.log_softmax(coarse_logits, dim=1)
        return self.gamma * (log_m.exp() * (log_m - log_c)).sum(dim=1).mean()
