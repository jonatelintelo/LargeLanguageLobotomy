"""Every change the pipeline makes to a target model's code, so that its router can be read and masked.

Trace collection (l3/traces.py) reads the experts a router selects with a forward hook on the module that
l3/models.py names as the gate, and silencing (l3/silence.py) masks the router logits with its own hook on that same
module. Most models need nothing; l3.models.load calls apply(), which installs what the others need. Each patch was
checked on the versions in requirements.txt (transformers 4.57.6) by running the model with and without it:

  model     patch          without it
  DeepSeek  dynamic_cache  its own model code calls DynamicCache methods that transformers 4.57 removed (seen_tokens,
                           get_max_length, get_usable_length): tracing, generation and scoring fail
            deepseek_gate  the stock gate returns (ids, weights, None) at inference, so silencing has no router logits
                           to mask; the patched gate returns the logits in place of the None
  Pangu     dynamic_cache  as for DeepSeek
  GPT-OSS   gptoss_router  the stock router returns a (scores, ids) tuple that neither hook can read; the patched router
                           returns its logits and the patched MLP selects the router's top-k experts from them, as the
                           stock MLP does. The stock experts module is kept: with this patch the output equals the
                           unpatched model's

The pipeline only runs the target model in eval mode, so the patched forwards implement the inference path only.
"""

import types

import torch
import torch.nn.functional as F


def apply(model, key):
    """Installs the patches the model registered as `key` in l3/models.py needs; most need none."""
    for patch in PATCHES.get(key, ()):
        patch(model)


def dynamic_cache(model):
    """Restores the DynamicCache methods that DeepSeek's and Pangu's own model code call. A change to the class, so it
    holds for every model loaded afterwards in the same process."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, input_seq_len, layer_idx=0: self.get_seq_length(layer_idx)


def deepseek_gate(model):
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            layer.mlp.gate.forward = types.MethodType(deepseek_moe_gate_forward, layer.mlp.gate)


def gptoss_router(model):
    for layer in model.model.layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
            layer.mlp.forward = types.MethodType(gptoss_mlp_forward, layer.mlp)
            layer.mlp.router.forward = types.MethodType(gptoss_router_forward, layer.mlp.router)


# The forward functions installed above.

def deepseek_moe_gate_forward(self, hidden_states):
    """DeepSeek-MoE's gate as upstream computes it at inference, returning the router logits in the place where the stock
    gate returns its auxiliary loss, which is None outside training. The target model is always in eval mode here."""
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    logits = F.linear(hidden_states, self.weight, None)
    return (*deepseek_select(self, logits), logits)


def deepseek_select(gate, logits):
    """(topk_idx, topk_weight) from a DeepSeek-MoE gate's logits, as upstream selects them; l3/silence.py reselects
    with this after masking."""
    if gate.scoring_func != "softmax":
        raise NotImplementedError(f"insupportable scoring function for MoE gating: {gate.scoring_func}")
    scores = logits.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=gate.top_k, dim=-1, sorted=False)
    if gate.top_k > 1 and gate.norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight


def gptoss_router_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    return router_logits


def gptoss_mlp_forward(self, hidden_states):
    router_logits = self.router(hidden_states)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


PATCHES = {"deepseek": (dynamic_cache, deepseek_gate), "pangu": (dynamic_cache,), "gptoss": (gptoss_router,)}
