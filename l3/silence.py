"""Expert silencing (Eq. 5).

Silencing local expert (l,e) replaces the router logit of e at layer l by negative infinity before
the softmax, so its selection probability is zero and the router redistributes the mass over the
experts that remain available:

    p' = softmax(g'),   g'_i = -inf if i == e else g_i

The intervention is a forward hook on the router module, so no weight is modified. Trace layer l is
the l-th router module of l3.models.gate_modules, the same list trace collection hooks, so a model
whose first decoder layer is dense maps correctly.

Most routers return their logits, and masking the returned tensor is enough because the selection
happens downstream. The DeepSeek gate is patched to return (indices, weights, logits), so its hook
masks the logits and selects from them again, as the patched gate does (l3.patches.deepseek_select).
"""

from collections import defaultdict

from l3 import patches

NEG_INF = float("-inf")


def _mask_logits_hook(expert_ids):
    def hook(module, inputs, output):
        masked = output.clone()
        masked[..., expert_ids] = NEG_INF
        return masked
    return hook


def _deepseek_hook(expert_ids):
    def hook(module, inputs, output):
        logits = output[2].clone()
        logits[..., expert_ids] = NEG_INF
        return (*patches.deepseek_select(module, logits), logits)
    return hook


def register(gates, spec, experts):
    """Silences `experts`, a list of (trace_layer, expert_id). Returns handles to remove."""
    by_layer = defaultdict(list)
    for layer, expert in experts:
        by_layer[int(layer)].append(int(expert))
    make = _deepseek_hook if spec.selection == "deepseek" else _mask_logits_hook
    handles = []
    for layer, ids in by_layer.items():
        assert 0 <= layer < len(gates), f"trace layer {layer} has no router module"
        ids = sorted(set(ids))
        assert len(ids) < spec.n_experts, (
            f"layer {layer}: silencing all {spec.n_experts} experts leaves the router nothing to "
            "select, which makes every routing weight NaN")
        handles.append(gates[layer].register_forward_hook(make(ids)))
    return handles


def remove(handles):
    for handle in handles:
        handle.remove()
