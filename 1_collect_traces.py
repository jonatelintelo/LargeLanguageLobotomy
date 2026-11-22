"""Stage 1: record token-resolved routing traces for one prompt set.

    python 1_collect_traces.py --model olmoe                                  # the training set
    python 1_collect_traces.py --model olmoe --set validation                 # or validation_same_source

Reads  the set's prompt file in data/ (see l3/data.py: SETS)
Writes artifacts/1_traces/<model>/traces_<set>.pkl    the traces
       artifacts/1_traces/<model>/meta_<set>.json      shapes and counts, no prompt text
       artifacts/1_traces/<model>/log_<set>.txt        this run's progress lines and summary
with <set> the set's name: training, validation or validation_same_source
"""

import argparse
import json
import time

from l3 import data, models, traces
from l3.log import log, text, to_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--set", default=data.TRAINING, choices=sorted(data.SETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="the first whole twin pairs totalling at most N prompts only (smoke test)")
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    prompt_file = data.SETS[args.set]
    to_file(data.traces_log_file(spec.name, args.set))
    questions, labels, groups = data.load_twin_set(prompt_file)
    sel = data.limit_by_group(groups, args.limit)
    if not sel:
        raise SystemExit(f"--limit {args.limit} selects no whole twin pair; use 2 or more")
    questions = [questions[i] for i in sel]
    labels = [labels[i] for i in sel]
    groups = [groups[i] for i in sel]
    log(f"{spec.name}: {len(questions)} prompts from {prompt_file}")

    t0 = time.time()
    model, tokenizer = models.load(args.model)
    log("model loaded, tracing")
    trace_list, traced, skipped, n_layers = traces.collect(model, tokenizer, spec, questions)
    if not trace_list:
        raise SystemExit("no prompt could be traced; the question span was never located")
    blob = {"traces": trace_list,
            "labels": [labels[i] for i in traced],
            "groups": [groups[i] for i in traced],
            # index into the set's prompt file, so later stages can look the prompt up
            "prompt_idx": [sel[i] for i in traced],
            "n_layers": n_layers}
    data.write_pickle(blob, data.traces_file(spec.name, args.set))

    meta = {"model": spec.name,
            "set": args.set,
            "prompt_file": prompt_file,
            "n_prompts_in": len(questions),
            "n_prompts_kept": len(trace_list),
            "n_prompts_skipped": len(skipped),
            "skipped_idx": [sel[i] for i in skipped],
            "n_layers": n_layers,
            "n_experts_per_layer": spec.n_experts,
            "top_k": spec.top_k,
            "n_local_experts": models.n_local_experts(spec, n_layers),
            "tokens_per_prompt": {"min": int(min(t.shape[0] for t in trace_list)),
                                  "max": int(max(t.shape[0] for t in trace_list))},
            "wall_clock_s": round(time.time() - t0, 1)}
    data.write_json(meta, data.traces_meta_file(spec.name, args.set))
    text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
