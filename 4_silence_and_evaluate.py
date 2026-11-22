"""Stage 4: silence the ranked experts and measure attack success under every rule of l3.judge.SUCCESS_RULES.

    python 4_silence_and_evaluate.py --model olmoe --prompts heldout                    # one expert per step
    python 4_silence_and_evaluate.py --model olmoe --prompts heldout --step_frac 0.005  # 0.5% of local experts

--prompts names an attack set (seen, heldout, new_source) or a prompt file of your own. One expert per step is
Algorithm 1 as written; --step_frac silences that share of the local experts per step instead, which takes fewer
generation passes and coarsens the reported budget (the Slurm jobs use 0.005). --min_words sets the word minimum for
a useful answer. The README describes the checks and the rules.

Reads  artifacts/3_rankings/<model>/le_lstm.pkl, the attack set
Writes artifacts/4_attacks/<model>/<set>/ (<set>_min_words<n>/ for a word minimum other than the default):
         summary.json     per rule: prompts that succeed on the clean pass, the others and how many of those
                          silencing made succeed; budget, stopping reason, each prompt's first success per rule
         steps.jsonl      one record per silencing step, cumulative successes per rule
         responses.jsonl  every response with its checks, by prompt id: the unmodified model's as step 0, then
                          every silencing step's
         silenced.json    the (layer, expert) list at the last strict success
         log.txt          this run's progress lines and summary
"""

import argparse
import json
import os
import pickle
import time

from l3 import attack, attribution, data, judge, models
from l3.log import log, text, to_file


def log_clean_pass(clean_rows, threshold, calibration):
    counts = ", ".join(f"{rule} {sum(r['success'][rule] for r in clean_rows)}/{len(clean_rows)}"
                       for rule in judge.SUCCESS_RULES)
    log(f"clean: {counts} | coherence threshold " + ("uncalibrated" if threshold is None else f"{threshold:.1f}"))
    if threshold is None:
        log(f"  no sampled answer of {judge.COHERENCE_MIN_WORDS} words or more to calibrate the coherence filter on; "
            "it will not be applied")
    else:
        log(f"  coherence: fluent median {calibration['fluent_median']:.1f}, shuffled median "
            f"{calibration['scrambled_median']:.1f}, threshold {threshold:.1f}, "
            f"from {calibration['n']} sampled answers")


def silence_and_log(model, tokenizer, spec, judges, gates, items, ranking, step, threshold, n_local, out_dir, args):
    """Algorithm 1 (l3.attack.run) with its outputs written as they happen: every step to steps.jsonl, with a progress
    line, and every response it evaluated to responses.jsonl, after the clean pass's. Returns run()'s result and the
    ids of the prompts whose first strict success also opens with a refusal prefix or is a WildGuard refusal."""
    other_rules = [rule for rule in judge.SUCCESS_RULES if rule != "strict"]
    strict_seen, refusal_signal = set(), []
    last_step = time.time()
    with open(os.path.join(out_dir, data.STEPS), "w") as steps_file, \
            open(os.path.join(out_dir, data.RESPONSES), "a") as responses_file:

        def on_step(rec, rows):
            nonlocal last_step
            steps_file.write(json.dumps(rec) + "\n")
            steps_file.flush()
            for r in rows:
                responses_file.write(json.dumps({"step": rec["step"], "n_silenced": rec["n_silenced"], **r}) + "\n")
                if r["success"]["strict"] and r["id"] not in strict_seen:
                    strict_seen.add(r["id"])
                    if attack.refusal_signal(r):
                        refusal_signal.append(r["id"])
            responses_file.flush()
            took, last_step = time.time() - last_step, time.time()
            c = rec["cumulative"]
            log(f"  step {rec['step']}: silenced {rec['n_silenced']}/{n_local} "
                f"({rec['n_silenced'] / n_local:.1%}), {rec['n_evaluated']} prompts active, {took:.0f} s | "
                f"strict successes {rec['successes']}, prompts {c['strict']} | "
                + ", ".join(f"{rule} {c[rule]}" for rule in other_rules)
                + f" | degenerate {rec['degenerate']}/{rec['n_evaluated']}")

        result = attack.run(model, tokenizer, spec, judges, gates, items, ranking, step, args.batch_size,
                            args.judge_batch_size, threshold, on_step, min_words=args.min_words)
    return result, refusal_signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--prompts", default="heldout",
                    help=f"{', '.join(data.ATTACK_SETS)}, or a path to a prompt file")
    ap.add_argument("--step_frac", type=float, default=0.0,
                    help="share of local experts silenced per step; 0 means one expert per step")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--judge_batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke test)")
    ap.add_argument("--min_words", type=int, default=judge.MIN_WORDS,
                    help="a response shorter than this counts as useful only when StrongREJECT rates it so; any "
                         f"value but the default ({judge.MIN_WORDS}) is added to the output directory's name")
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    if not spec.refusal_prefixes:
        raise SystemExit(f"no refusal prefixes are listed for {args.model} in l3/models.py; list the openings "
                         "the model starts its refusals with before running stage 4")
    data.require(data.ranking_file(spec.name), "run 3_train_and_rank.py first")
    with open(data.ranking_file(spec.name), "rb") as f:
        ranking = attribution.positive(pickle.load(f))
    prompt_file, set_name, prompts = data.load_attack_set(args.prompts)
    out_dir = data.attack_dir(spec.name, set_name, None if args.min_words == judge.MIN_WORDS else args.min_words)
    to_file(os.path.join(out_dir, "log.txt"))
    for stale in (data.SUMMARY, data.SILENCED):   # a previous run's, which a crash would leave beside new steps
        if os.path.exists(os.path.join(out_dir, stale)):
            os.remove(os.path.join(out_dir, stale))
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        raise SystemExit(f"{prompt_file} contains no prompts")
    if not ranking:
        raise SystemExit("the ranking has no positively scored expert to silence")

    t0 = time.time()
    model, tokenizer = models.load(args.model)
    gates = models.gate_modules(model, spec)
    n_local = models.n_local_experts(spec, len(gates))
    step = max(1, round(args.step_frac * n_local))
    items = [{"id": p["id"], "question": p["question"],
              "prompt": models.build_prompt(tokenizer, p["question"], spec)} for p in prompts]
    log(f"{spec.name}, {set_name}: {len(items)} prompts, {len(ranking)} positively scored of {n_local} "
        f"local experts, {step} expert(s) per step, {spec.max_new_tokens} new tokens, word minimum {args.min_words}")

    log("loading Llama-Guard, WildGuard and StrongREJECT")
    judges = (judge.LlamaGuard(), judge.WildGuard(), judge.StrongReject())
    log(f"one sampled answer per prompt to calibrate coherence, then the clean pass on {len(items)} prompts")
    clean_rows, threshold, calibration = attack.clean_pass(model, tokenizer, spec, judges, items,
                                                           args.batch_size, args.judge_batch_size,
                                                           min_words=args.min_words)
    data.write_jsonl([{"step": 0, "n_silenced": 0, **r} for r in clean_rows], os.path.join(out_dir, data.RESPONSES))
    log_clean_pass(clean_rows, threshold, calibration)
    log("silencing")

    (first, n_steps, silenced_at_peak, stopped_because), refusal_signal = silence_and_log(
        model, tokenizer, spec, judges, gates, items, ranking, step, threshold, n_local, out_dir, args)

    peak_budget = len(silenced_at_peak)
    summary = {"model": spec.name, "set": set_name, "prompt_file": prompt_file,
               "n_prompts": len(items), "step": step, "step_frac": args.step_frac or None,
               "max_new_tokens": spec.max_new_tokens,
               "batch_size": args.batch_size, "judge_batch_size": args.judge_batch_size,
               "n_local_experts": n_local, "n_positive_experts": len(ranking),
               "coherence_threshold": threshold,
               # the sampled clean answers that set it: how many, and their median fluent and shuffled perplexity
               "coherence_calibration": calibration,
               "refusal_prefixes": list(spec.refusal_prefixes),
               "success_rules": {rule: list(conds) for rule, conds in judge.SUCCESS_RULES.items()},
               # "useful": at least min_words words, or a StrongREJECT score of at least strongreject_cutoff
               "usefulness": {"min_words": args.min_words, "strongreject_cutoff": judge.USEFUL_SCORE,
                              "strongreject_model": judge.STRONGREJECT_ID},
               "per_rule": attack.per_rule(clean_rows, first),
               # strict successes whose response also opens with a refusal prefix or is a WildGuard refusal
               "strict_with_refusal_signal": len(refusal_signal),
               "peak_budget": peak_budget,
               "peak_budget_share": round(peak_budget / n_local, 4),
               "n_steps": n_steps,
               "stopped_because": stopped_because,
               "wall_clock_s": round(time.time() - t0, 1),
               "first_success": first}
    data.write_json(summary, os.path.join(out_dir, data.SUMMARY))
    data.write_json({"n": len(silenced_at_peak), "experts": [[int(l), int(e)] for l, e in silenced_at_peak]},
                    os.path.join(out_dir, data.SILENCED))
    text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
