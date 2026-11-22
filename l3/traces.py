"""Routing-trace collection (Eq. 1).

For a prompt of T tokens and L routed layers, we record which experts the router selects for every
token and layer. The stored trace is an integer array of shape (T, L, K) holding expert ids; the
multi-hot vector x_t of Eq. 1 is built from it at training time, in l3.lelstm.

Two properties this module guarantees, both of which matter for the twin comparison:

  * The trace is a function of the prompt alone. Prompts are run one at a time, so no padding enters
    the forward pass and position ids are the prompt's own. In a left-padded batch, the routing a
    model produces for the same text can differ.
  * T counts the tokens of the request only. Chat-template, system and generation-prompt tokens are
    located through the tokenizer's offset mapping and excluded, so a twin pair differs in the
    tokens that carry intent rather than in template scaffolding.
"""

import numpy as np
import torch

from l3 import models
from l3.log import log

PROGRESS_EVERY = 100   # prompts between progress lines


def span_from_offsets(prompt_text, question_text, offsets):
    """Token index range of the question, from a fast tokenizer's character offsets."""
    needle = question_text.strip()
    char_start = prompt_text.rfind(needle)
    if char_start == -1:
        return None, None
    char_end = char_start + len(needle)
    start = end = None
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:  # special token
            continue
        if start is None and tok_end > char_start:
            start = i
        if tok_start < char_end:
            end = i
    return start, end


def span_from_ids(prompt_ids, question_ids):
    """Token index range of the question, by locating its id subsequence.

    Used for a slow tokenizer, which cannot return offsets.
    """
    n = len(question_ids)
    for i in range(len(prompt_ids) - n + 1):
        if prompt_ids[i:i + n] == question_ids:
            return i, i + n - 1
    return None, None


def question_span(tokenizer, prompt_text, question_text, enc):
    """Token index range of `question_text` inside the templated prompt, inclusive.

    Returns (None, None) if the question cannot be located, and raises if a located span does not
    decode back to the question, so a silent mislocation cannot reach the training data.
    """
    ids = enc["input_ids"][0].tolist()
    if tokenizer.is_fast:
        start, end = span_from_offsets(prompt_text, question_text,
                                       enc["offset_mapping"][0].tolist())
    else:
        start, end = span_from_ids(ids, tokenizer(question_text, add_special_tokens=False)["input_ids"])
    if start is None:
        return None, None
    # Verified ignoring whitespace: decoding is not always byte-exact on spacing, but a span that
    # covers the wrong tokens fails this check.
    recovered = tokenizer.decode(ids[start:end + 1], skip_special_tokens=True)
    assert "".join(recovered.split()) == "".join(question_text.split()), \
        f"span {start}:{end} does not decode back to the question"
    return start, end


@torch.inference_mode()
def collect(model, tokenizer, spec, questions):
    """Returns (traces, traced, skipped, n_layers): traces[i] has shape (T_i, L, K) and belongs to questions[traced[i]];
    `skipped` lists the questions whose span could not be located."""
    gates = models.gate_modules(model, spec)
    n_layers = len(gates)
    current = {}

    def make_hook(trace_layer):
        def hook(module, inp, output):
            current[trace_layer] = models.selected_experts(output, spec).detach().cpu().numpy()
        return hook

    handles = [module.register_forward_hook(make_hook(l)) for l, module in enumerate(gates)]
    traces, traced, skipped = [], [], []
    try:
        for i, question in enumerate(questions):
            prompt = models.build_prompt(tokenizer, question, spec)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                            return_offsets_mapping=tokenizer.is_fast)
            start, end = question_span(tokenizer, prompt, question, enc)
            enc.pop("offset_mapping", None)
            if start is None:
                skipped.append(i)
                continue
            current.clear()
            model(**{k: v.to(model.device) for k, v in enc.items()})
            assert len(current) == n_layers, f"{len(current)} of {n_layers} routers fired"
            # (L, tokens, K) -> (tokens, L, K), then the request's own tokens
            trace = np.stack([current[l] for l in range(n_layers)], axis=1)[start:end + 1]
            assert trace.shape[1] == n_layers and trace.shape[2] == spec.top_k, trace.shape
            assert trace.max() < spec.n_experts, f"expert id {trace.max()} >= {spec.n_experts}"
            traces.append(trace.astype(np.int16 if spec.n_experts <= 2 ** 15 else np.int32))
            traced.append(i)
            if (i + 1) % PROGRESS_EVERY == 0:
                log(f"  {i + 1}/{len(questions)} prompts")
    finally:
        for h in handles:
            h.remove()
    return traces, traced, skipped, n_layers
