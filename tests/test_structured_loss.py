"""The reusable takeaway module must work on ANY two-level taxonomy, not just
CIFAR-100. Test it on a tiny hand-built tree where we can check every number."""

import torch

from structured_loss import HeadAgreementLoss, SoftHierarchyLoss, build_parent_matrix

# 4 children under 2 parents: children 0,1 -> parent 0 ; children 2,3 -> parent 1.
CHILD_TO_PARENT = [0, 0, 1, 1]


def test_build_parent_matrix():
    m = build_parent_matrix(CHILD_TO_PARENT)
    assert m.shape == (2, 4)
    assert m.tolist() == [[1, 1, 0, 0], [0, 0, 1, 1]]


def test_marginalized_mass_and_loss_value():
    loss_fn = SoftHierarchyLoss(CHILD_TO_PARENT)
    # Put ~all mass on child 0 (parent 0).
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
    good = loss_fn(logits, torch.tensor([0]))   # true parent 0 -> mass ~1 -> ~0
    bad = loss_fn(logits, torch.tensor([1]))    # true parent 1 -> mass ~0 -> large
    assert good.item() < 1e-3
    assert bad.item() > 5.0


def test_differentiable():
    loss_fn = SoftHierarchyLoss(CHILD_TO_PARENT)
    logits = torch.randn(3, 4, requires_grad=True)
    parents = torch.tensor([0, 1, 0])
    loss = loss_fn(logits, parents)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_violation_rate():
    loss_fn = SoftHierarchyLoss(CHILD_TO_PARENT)
    # argmax children: 0 (parent 0) and 2 (parent 1).
    logits = torch.tensor([[10.0, 0, 0, 0], [0, 0, 10.0, 0]])
    # Compared against parents [0, 0]: first consistent, second violates.
    assert loss_fn.violation_rate(logits, torch.tensor([0, 0])) == 50.0
    assert loss_fn.violation_rate(logits, torch.tensor([0, 1])) == 0.0


def test_head_agreement_zero_when_heads_agree_positive_when_not():
    loss_fn = HeadAgreementLoss(CHILD_TO_PARENT)
    fine = torch.tensor([[10.0, -10.0, -10.0, -10.0]])        # mass in parent 0
    agree = loss_fn(torch.tensor([[10.0, -10.0]]), fine)
    clash = loss_fn(torch.tensor([[-10.0, 10.0]]), fine)
    assert agree.item() < 1e-3
    assert clash.item() > 5.0
