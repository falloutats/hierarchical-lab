# hierarchical-lab

> **When you know the structure among your outputs, build it into the architecture *and* the objective.**

CIFAR-100's 100 fine classes nest under 20 superclasses, which gives you a real taxonomy to exploit. This repo
is a staged walkthrough. Each script changes one thing, measures it against the previous stage, and runs in
seconds per epoch on Apple-Silicon MPS.

**Built on [Ugenteraan/Deep_Hierarchical_Classification](https://github.com/Ugenteraan/Deep_Hierarchical_Classification)**
by Ugenteraan Manogaran (MIT, © 2021), an unofficial implementation of
[Wan et al. 2020, *Deep Hierarchical Classification for Category Prediction in E-commerce*](https://arxiv.org/abs/2005.06692).
The coarse-to-fine head and the hierarchical losses here follow that repo. The critique below is a good-faith
technical note about one implementation detail. See [`NOTICE`](NOTICE).

## TL;DR

| finding | evidence |
|---|---|
| The upstream dependency loss has **no gradient**. It is built from `argmax`/`==`/`where`. | `requires_grad=False`. Training with and without it is bit-for-bit identical. |
| A correct, differentiable replacement (`SoftHierarchyLoss`) **did not help**. | Violation rate went from 26.39% to 28.07%. |
| Putting the measured quantity in the loss (`HeadAgreementLoss`) **did help**. | Violation rate went from 26.39% to 20.52%, with fine accuracy within 0.4 pts. |
| Coarse-to-fine conditioning did **not** beat plain two-head training at this scale. | Fine accuracy 48.16% vs 50.79%. |

## The ideas

**1. Coarse-to-fine conditioning (architecture).** A shared backbone feeds a coarse head (20 logits). Those
logits are concatenated into the fine head: `cat(20, 100) → 100`. This is `models.CoarseToFineModel`, faithful to
the upstream `model/resnet50.py`. One caveat: the coarse logits are themselves linear in the features, so this
coupling reparameterizes the fine head without making it more expressive. See the model's docstring.

**2. Taxonomy in the loss.**

- **Layer loss** (`lloss`) is cross-entropy at every level, summed. `losses.layer_loss`
- **Dependency loss** penalizes a fine prediction that isn't a child of the predicted superclass.

## The dead loss

The upstream dependency loss:

```
D      = 1 if argmax(fine) is not a child of argmax(coarse) else 0
l_prev = 0 if argmax(coarse) == coarse_true else 1
l_curr = 0 if argmax(fine)   == fine_true   else 1
dloss  = Σ p^(D·l_prev) · p^(D·l_curr) − 1          # p = 3
```

Every term is piecewise-constant in the logits. A small change to a logit doesn't move the argmax, so the
derivative is zero almost everywhere. Autograd doesn't even build a graph for it. The term can log a value of
~790 and still contribute **zero gradient**. It works as a monitoring number, not as a loss.

**The way out is to measure mass, not the winner.** Let `M` be the `20 × 100` membership matrix, with
`M[c, f] = 1` iff `f` is a child of `c`. Then `probs @ Mᵀ` is the vector of parent marginals. Marginalizing up
a tree is a matrix product, and a matrix product is smooth in every logit. Both replacement losses build on this.
They are computed in log space (`log_softmax` → `masked_fill(-inf)` → `logsumexp`), so they stay exact when a
parent's mass underflows float32. A test pins that case.

## How the fix was found

1. **Predicted** that `dloss` carries no gradient, then **confirmed** it two ways. It raises on `.backward()`, and
   runs with and without it from the same seed are identical at every epoch.
2. **Tried `SoftHierarchyLoss` = −log(fine mass inside the *true* parent).** It is differentiable and reaches the
   backbone, but the violation rate got slightly *worse*.
3. **Diagnosed why.** The loss compares the fine head with the **label**. The metric compares the fine head with
   the **coarse head**, and the label never appears in it. The loss is also implied by the fine cross-entropy
   that `lloss` already contains: `p(fine_true) ≤ mass(parent_true)`. So it added almost nothing new.
4. **Wrote `HeadAgreementLoss` = KL(m ‖ c).** Here `m` is the fine distribution marginalized to the parents, and
   `c` is the coarse head's distribution. With the measured quantity in the objective, the violation rate fell.

**Lesson:** a regularizer can be differentiable, correct, and well-motivated and still do nothing for your
metric if it doesn't optimize what you measure. You need a controlled A/B and a metric separate from accuracy
to see that.

## Results

All numbers come from real runs: Apple M-series MPS, seed 0, and identical initialization and data order
across variants.

**Stage 4: four objectives, 6 epochs.** Violation % is how often the fine prediction's parent disagrees with
the coarse head.

| objective | fine acc | violation % |
|---|---|---|
| `lloss` only | 43.25% | 26.39% |
| `lloss` + upstream `dloss` | 43.25% | 26.39% (identical, as predicted) |
| `lloss` + `SoftHierarchyLoss` | 40.95% | 28.07% |
| `lloss` + `HeadAgreementLoss` | 42.88% | **20.52%** |

**Stages 1–3: 8 epochs.**

| stage | fine acc | coarse acc | violation % |
|---|---|---|---|
| 1 · flat baseline | 49.46% | 63.31% | 36.69%\* |
| 2 · two heads + `lloss` | 50.79% | 62.86% | 23.08% |
| 3 · coarse-to-fine conditioning | 48.16% | 60.42% | 24.17% |

\* Stage 1 has one head, so it has nothing to disagree with. Its violation rate is measured against the *true*
superclass (= 100 − coarse acc) and is **not comparable** with stages 2–3.

Stage 3 did not reproduce the upstream gain. This run used a small CNN for 8 epochs, not ResNet50 for 100, so
treat the result as "no gain at this scale" rather than a verdict.

## The pathway

| stage | script | idea | output in `runs/` |
|---|---|---|---|
| 0 | `s0_tree.py` | see the taxonomy | `hierarchy_tree.png` |
| 1 | `s1_flat_baseline.py` | flat 100-way classifier and the violations it implies | `s1_confusion.png`, `s1_curves.png` |
| 2 | `s2_two_head_lloss.py` | coarse + fine heads, layer loss | `s2_confusion.png`, `s2_curves.png` |
| 3 | `s3_coarse_to_fine.py` | feed coarse logits into the fine head | `s3_confusion.png`, `s3_curves.png` |
| 4 | `s4_dependency_loss.py` | prove `dloss` is dead, then find a loss that works | `violation_comparison.png` |
| 5 | `s5_representation_evolution.py` | watch features cluster by superclass | `embedding_evolution.gif`, `embedding_trajectory.png` |
| — | `structured_loss.py` | reusable, dataset-agnostic losses | — |

## Run it

```bash
uv sync
uv run python s0_tree.py                        # instant, no download
uv run python s1_flat_baseline.py --epochs 8
uv run python s4_dependency_loss.py --epochs 6  # the main experiment
uv run python s5_representation_evolution.py --epochs 12
uv run pytest
```

The training scripts accept `--epochs --lr --batch-size --seed --device`.

**On macOS**, torchvision's downloader may fail with `CERTIFICATE_VERIFY_FAILED`. If it does, fetch the archive
once with curl and re-run:

```bash
mkdir -p data
curl -C - -fSL -o data/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```

## Changes from upstream

- **Device:** MPS first, then CUDA, then CPU (`device.pick_device`). Upstream silently falls back to CPU on a Mac.
- **Data:** `torchvision.datasets.CIFAR100` plus a hard-coded fine→coarse map (`hierarchy.py`). This replaces
  the upstream PNG dump, CSV, and cv2 re-read.
- **Visuals:** a taxonomy tree, a confusion matrix grouped into superclass blocks, violation curves, and a GIF
  of the PCA embedding over training. Upstream has three line charts.
- **Comparisons:** every variant shares a seed and data order, so each difference comes from one change.

## Where this transfers

Any model whose outputs have known structure can use this: label taxonomies in LLM routing or classification,
coarse-to-fine decoders, product categorization, ICD codes and other ontologies. Point
`structured_loss.SoftHierarchyLoss` / `HeadAgreementLoss` at any `child_to_parent` list.

**Enforce structure by marginalizing probabilities, never by argmax.**
