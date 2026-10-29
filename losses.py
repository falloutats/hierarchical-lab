"""The hierarchy losses — where the taxonomy meets the optimizer.

Four objects live here:

  1. layer_loss        their `lloss`: per-level cross-entropy. Healthy.
  2. faithful_dloss    their `dloss`, ported faithfully. Kept ONLY as evidence:
                       it is piecewise-constant and therefore carries no gradient.
  3. SoftHierarchyLoss a first differentiable replacement, built on the
                       observation that "does the child respect its parent?" is
                       a MARGINALIZATION — a linear map — not a comparison.
                       Correct, but it did not reduce head disagreement.
  4. HeadAgreementLoss the replacement that did: the same marginalization, but
                       compared against the coarse head instead of the label.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hierarchy import coarse_of, parent_matrix


# ----------------------------------------------------------------------------
# 1. Layer loss
# ----------------------------------------------------------------------------
def layer_loss(coarse_logits, fine_logits, coarse_true, fine_true, alpha: float = 1.0):
    """L = alpha * ( CE(coarse) + CE(fine) ).

    Cross-entropy here is the negative log-likelihood of the true class under the
    softmax distribution: CE = -log p(y_true). Adding the two levels is not an
    arbitrary "multi-task weighting" — it is the log of a product. Maximizing
    log p(coarse) + log p(fine) maximizes p(coarse) * p(fine), i.e. we are fitting
    the two levels as if they were independent. That independence assumption is
    precisely what is wrong with this objective on its own, and precisely the gap
    the dependency loss below is trying (and failing) to close.
    """
    return alpha * (F.cross_entropy(coarse_logits, coarse_true)
                    + F.cross_entropy(fine_logits, fine_true))


# ----------------------------------------------------------------------------
# 2. Their dependency loss — the cautionary tale
# ----------------------------------------------------------------------------
def faithful_dloss(coarse_logits, fine_logits, coarse_true, fine_true,
                   beta: float = 0.8, p_loss: float = 3.0):
    """Reproduces the upstream dependency loss so we can dissect it.

    The intent is right: charge the network extra when the fine prediction is not
    a child of the coarse prediction. The execution kills it. Every quantity below
    is obtained by argmax or by an equality test, and both are *piecewise-constant*
    functions of the logits: nudge a logit by a hair and the argmax does not move,
    so the derivative is 0 almost everywhere (and undefined at the jumps). A
    function that is flat almost everywhere transmits no information to gradient
    descent. Autograd reports this honestly by never building a graph node at all,
    so the returned tensor has requires_grad=False and cannot even .backward().

    Two pieces of algebra worth seeing:

      p^(D*l_prev) * p^(D*l_curr) = p^(D*(l_prev + l_curr))

    so the penalty only depends on D times the NUMBER of wrong levels: it takes
    the values {0, 2, 8} for p=3, namely p^0-1, p^1-1, p^2-1. And note D=1 forces
    at least one level to be wrong: if both predictions were correct, the fine
    truth's parent is the coarse truth by construction, so the pair would be
    consistent and D would be 0. The penalty is therefore only ever charged on
    top of an already-incorrect prediction.
    """
    coarse_pred = torch.argmax(coarse_logits, dim=1)
    fine_pred = torch.argmax(fine_logits, dim=1)
    # Softmax is monotonic, so argmax(softmax(z)) == argmax(z); the upstream code
    # applies softmax first, which changes nothing but the runtime.

    # D = 1 when the predicted child does not sit under the predicted parent.
    # Because the taxonomy is a partition (every fine class has exactly ONE
    # parent), "is fine_pred among the children of coarse_pred" reduces to a
    # single table lookup instead of the upstream per-sample Python membership
    # test over a dict of lists.
    D_l = (coarse_of(fine_pred) != coarse_pred).float()

    l_prev = torch.where(coarse_pred == coarse_true, 0.0, 1.0)
    l_curr = torch.where(fine_pred == fine_true, 0.0, 1.0)

    dloss = torch.sum(torch.pow(p_loss, D_l * l_prev) * torch.pow(p_loss, D_l * l_curr) - 1.0)
    return beta * dloss


# ----------------------------------------------------------------------------
# 3. The differentiable replacement
# ----------------------------------------------------------------------------
class SoftHierarchyLoss(nn.Module):
    """L = -log( sum of fine probability mass sitting inside the TRUE superclass ).

    The idea that makes this differentiable: stop asking "which class won?" and
    start asking "how much probability landed in the right block?". The first is
    a step function; the second is smooth in every logit.

    The linear algebra. Let M be the (20 x 100) membership matrix with
    M[c, f] = 1 exactly when fine class f is a child of coarse class c. Each
    column holds a single 1 (one parent per child), so M is the one-hot encoding
    of parenthood. For a probability row-vector p over the 100 fine classes,

        p @ M.T   ->   a 20-vector whose c-th entry is  sum_{f in children(c)} p_f

    which is exactly the marginal probability of the parent. So marginalizing a
    distribution up a tree IS a matrix product against the membership matrix; the
    tree structure lives entirely in M. Grouping/aggregation and matrix
    multiplication are the same operation here.

    Why compute it in log-space instead of literally doing `probs @ M.T`. The
    quantity we want is

        -log sum_{f in children(c)} exp(z_f) / sum_j exp(z_j)

    Forming the probabilities first, summing, then taking a log invites underflow:
    if the mass in the parent is ~1e-45 the sum flushes to 0 and the log returns
    -inf (the usual patch is to add an epsilon, which silently caps the loss and
    biases the gradient). Instead we take log_softmax, which computes
    z - logsumexp(z) with the max subtracted internally, then reduce the children
    with torch.logsumexp. logsumexp(x) = max(x) + log sum exp(x - max(x)) never
    overflows and stays exact deep into the tail, so no epsilon is needed and the
    gradient is right even when the model is confidently wrong -- which is exactly
    when this loss has the most to say.

    We select the children with masked_fill(-inf) rather than multiplying by M:
    exp(-inf) = 0 contributes nothing to the sum, whereas multiplying probabilities
    by a 0/1 mask would put a hard zero *inside* a log.

    A caveat the experiments bear out. The true fine class is one of the children
    of the true parent, so its probability is part of the summed mass:

        p(fine_true)  <=  mass(parent_true)   =>   -log mass  <=  CE(fine)

    This loss is therefore a strictly weaker version of the fine cross-entropy
    that layer_loss already contains: whenever the fine head is right, this term
    is already small. It adds little new pressure, and none at all on the coarse
    head -- see HeadAgreementLoss for what that implies.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        # register_buffer (not a plain attribute) so M rides along on .to(device)
        # and is saved in state_dict, without ever being treated as a parameter.
        self.register_buffer("M", parent_matrix())

    def forward(self, fine_logits: torch.Tensor, coarse_true: torch.Tensor) -> torch.Tensor:
        log_p_fine = F.log_softmax(fine_logits, dim=1)

        # Row c of M indicates c's children, so indexing M by the batch of true
        # parents broadcasts a (B, 100) sibling mask in one gather -- no loop.
        sibling_mask = self.M[coarse_true]
        masked = log_p_fine.masked_fill(sibling_mask == 0, float("-inf"))

        log_mass_in_parent = torch.logsumexp(masked, dim=1)
        return self.beta * (-log_mass_in_parent).mean()


class HeadAgreementLoss(nn.Module):
    """L = KL( fine-marginalized parent distribution || coarse head distribution ).

    Why this exists, and why SoftHierarchyLoss was not enough. Measured on
    CIFAR-100, adding SoftHierarchyLoss did NOT reduce the head-disagreement rate
    (see the Results table in the README). The reason is a mismatch between what
    that loss optimizes and what `violation_rate` measures:

        SoftHierarchyLoss  ties the fine distribution to the TRUE parent label.
        violation_rate     compares the fine prediction's parent against the
                           COARSE HEAD's prediction -- the truth never appears.

    So the first loss can be fully satisfied while the two heads still contradict
    each other. To move a metric you must put that metric's own quantity in the
    objective. Here both distributions over the 20 parents are compared directly:

        m = marginalized fine distribution   (fine probs summed within each parent)
        c = coarse head's own distribution
        L = KL(m || c) = sum_p m_p * (log m_p - log c_p)

    KL is not symmetric, and the direction is a modelling choice. KL(m || c)
    weights each term by m, so it is largest where the FINE head puts mass on a
    parent the coarse head considers unlikely. Gradient flows into both heads,
    but they feel it differently:

      * coarse head: d/d(coarse logits) = c - m, the same gradient as
        cross-entropy with m as a soft target. The coarse head is pulled toward
        the fine head's (100-way, more specific) opinion.
      * fine head: as a function of m, KL(m || c) is mode-seeking -- it is cheap
        to drop mass from a parent but expensive to put mass where c is near 0.
        The fine head is pushed to keep its mass inside parents the coarse head
        already believes in.

    So the heads meet rather than one chasing the other. Agreeing on a wrong
    answer would satisfy this term, which is why it is only ever added to
    layer_loss, never used alone.

    Everything is computed from log-probabilities for the same reason as above:
    m_p is a sum over children, which is a logsumexp in log-space, and KL needs
    log m and log c anyway. Forming probabilities first would only add underflow.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("M", parent_matrix())

    def forward(self, coarse_logits: torch.Tensor, fine_logits: torch.Tensor) -> torch.Tensor:
        log_p_fine = F.log_softmax(fine_logits, dim=1)                    # (B, 100)

        # Marginalize to ALL parents at once. Broadcasting the (20, 100) mask over
        # the batch gives (B, 20, 100); -inf on non-children drops them from the
        # reduction, so logsumexp over the last axis is a per-parent group-sum
        # carried out in log-space.
        masked = log_p_fine.unsqueeze(1).masked_fill(self.M.unsqueeze(0) == 0, float("-inf"))
        log_m = torch.logsumexp(masked, dim=2)                            # (B, 20)

        log_c = F.log_softmax(coarse_logits, dim=1)                       # (B, 20)
        kl = (log_m.exp() * (log_m - log_c)).sum(dim=1)
        return self.gamma * kl.mean()


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
@torch.no_grad()
def accuracy(logits, target) -> float:
    return (logits.argmax(1) == target).float().mean().item() * 100.0


@torch.no_grad()
def violation_rate(coarse_logits, fine_logits) -> float:
    """% of samples where the two heads contradict each other about the taxonomy.

    This is the model's INTERNAL consistency: it compares the parent implied by
    the fine prediction against the coarse head's own prediction, ignoring the
    ground truth entirely. A model can be inconsistent while both heads are wrong,
    or consistent while both are wrong together. Kept separate from accuracy for
    that reason -- it is the quantity the dependency loss claims to minimize.
    """
    return (coarse_of(fine_logits.argmax(1)) != coarse_logits.argmax(1)).float().mean().item() * 100.0


@torch.no_grad()
def flat_violation_rate(fine_logits, coarse_true) -> float:
    """Violation measured against the TRUE parent, for models with no coarse head.

    Stage 1 has only one head, so there is no second opinion to contradict; we ask
    instead whether the predicted fine class at least lands in the correct
    superclass. That makes it exactly 100 - (implied coarse accuracy): it measures
    correctness against the truth, whereas violation_rate measures agreement
    between two heads. Different questions, so the numbers are not comparable.
    """
    return (coarse_of(fine_logits.argmax(1)) != coarse_true).float().mean().item() * 100.0
