"""Gradient x input attribution and the ranked expert list (Eqs. 2-4).

For the refusal logit z of a trained classifier and the multi-hot input x of Eq. 1,

    s_{t,l,e} = (dz / dx_t[(l,e)]) * x_t[(l,e)]                                     (Eq. 2)
    S^p_{l,e} = sum over the tokens of prompt p that used (l,e) of s_{t,l,e}        (Eq. 3)
    S_{l,e}   = sum over refused malicious training prompts p of S^p_{l,e}          (Eq. 4)

The logit is differentiated rather than the probability, since the sigmoid saturates on confidently
classified prompts and would flatten exactly the traces the ranking depends on. Because x is an
indicator, the product keeps the gradient at the experts that processed the token and zeroes the
rest, so a local expert is credited only on the tokens it actually processed.

The ranking is every local expert sorted by S_{l,e} descending; positive() gives the positively scored ones,
which stage 4 silences. The refused malicious prompts are the training prompts labelled 1; stage 3 also trains on
malicious prompts the model answered, labelled 0 like the benign twins, and does not attribute those.

With an ensemble, each member's S is divided by its own standard deviation over all local experts and
the scaled scores are averaged. Scaling equalizes the members' weight in the average without moving
zero, so a positive mean still means the expert raises the refusal logit; centering the scores instead
would turn "positive" into "above the average expert". The order is the same either way.
"""

import itertools
from collections import defaultdict

import numpy as np
import torch

from l3 import lelstm

ATTRIBUTION_BATCH_SIZE = 16
AGREEMENT_STEP = 0.5   # percent of the local experts between the points of the member-agreement curve


def token_attributions(model, traces, idx, device):
    """Eq. 2 for every prompt in `idx`. Returns {prompt index: array of shape (T, L, K)}."""
    model.to(device).eval()
    out = {}
    for s in range(0, len(idx), ATTRIBUTION_BATCH_SIZE):
        batch = list(idx[s:s + ATTRIBUTION_BATCH_SIZE])
        ids, lengths = lelstm.collate(traces, batch)
        ids = ids.to(device)
        x = model.multi_hot(ids).detach().requires_grad_(True)
        # summing over the batch is safe: samples are independent, so d(sum z)/dx[b] = dz_b/dx[b]
        with torch.backends.cudnn.flags(enabled=False):
            model.forward_from_multi_hot(x, lengths).sum().backward()
        grad = x.grad.reshape(ids.shape[0], ids.shape[1], model.n_layers, model.n_experts)
        # s = grad at the selected (l,e); x is 1 there and 0 elsewhere
        s = torch.gather(grad, 3, ids.long()).detach().cpu().numpy()
        for j, i in enumerate(batch):
            out[i] = s[j, : int(lengths[j])]
    return out


def aggregate(token_attr, traces):
    """Eqs. 3-4: sum the per-token scores into one score per local expert (l,e)."""
    total = defaultdict(float)
    for i, s in token_attr.items():
        n_tokens, n_layers, top_k = s.shape
        layers = np.broadcast_to(np.arange(n_layers)[None, :, None], (n_tokens, n_layers, top_k))
        for l, e, v in zip(layers.ravel(), traces[i].ravel(), s.ravel()):
            total[(int(l), int(e))] += float(v)
    return total


def positive(ranked):
    """The prefix of the ranking with a positive score: the experts that are silenced."""
    return [(l, e) for (l, e), score in ranked if score > 0]


def as_vector(scores, n_layers, n_experts):
    """Scores as a flat array over all local experts in (layer, expert) order; unselected experts are zero."""
    v = np.zeros(n_layers * n_experts)
    for (l, e), score in scores.items():
        v[l * n_experts + e] = score
    return v


def ensemble_ranking(member_scores, n_layers, n_experts):
    """Scale each member's scores by their standard deviation, average them, and rank all local experts."""
    scaled = []
    for scores in member_scores:
        v = as_vector(scores, n_layers, n_experts)
        sd = v.std()
        scaled.append(v / sd if sd > 0 else v)
    mean = np.mean(scaled, axis=0)
    ranked = [((l, e), float(mean[l * n_experts + e])) for l in range(n_layers) for e in range(n_experts)]
    return sorted(ranked, key=lambda kv: kv[1], reverse=True)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def chance_jaccard(share):
    """The expected Jaccard overlap of two random sets that each hold `share` of the same population."""
    return share / (2 - share)


def member_agreement(member_scores, n_layers, n_experts):
    """How closely the members' own rankings agree: the Jaccard overlap between every pair of members of the expert
    sets each selects on its own, as {"mean", "min", "max"} over the pairs. `curve` takes the top x% of each
    member's ranking for x = AGREEMENT_STEP, 2 * AGREEMENT_STEP, ..., 100; `positive` takes each member's positively
    scored experts, the set it would silence on its own, with the mean share of local experts that set holds. None
    with fewer than two members."""
    if len(member_scores) < 2:
        return None
    own = [ensemble_ranking([s], n_layers, n_experts) for s in member_scores]

    def spread(sets):
        js = [jaccard(a, b) for a, b in itertools.combinations(sets, 2)]
        return {"mean": round(float(np.mean(js)), 4), "min": round(min(js), 4), "max": round(max(js), 4)}

    n = n_layers * n_experts
    top_percent = [round(AGREEMENT_STEP * i, 2) for i in range(1, round(100 / AGREEMENT_STEP) + 1)]
    curve = [spread([[e for e, _ in r[:max(1, round(p / 100 * n))]] for r in own]) for p in top_percent]
    mine = [positive(r) for r in own]
    return {"top_percent": top_percent, "curve": curve,
            "positive": {**spread(mine), "share": round(float(np.mean([len(m) for m in mine])) / n, 4)}}

