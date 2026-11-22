"""Stage 2: per-model behavioral filter, which keeps a prompt only where the unmodified model does what its label says.

    python 2_filter_training_data.py --model olmoe                    # the training set
    python 2_filter_training_data.py --model olmoe --set validation   # or validation_same_source

The unmodified model answers every traced prompt; Llama-Guard and WildGuard judge each response, and
l3.judge.filter_decision keeps a malicious prompt only when the model refuses it harmlessly and a benign twin only
when the model answers it harmlessly. Validation sets are filtered the same way, so stage 3 scores the classifier
against labels that match the model's own behavior. The README explains the rule.

Reads  artifacts/1_traces/<model>/traces_<set>.pkl and the set's prompt file
Writes, in artifacts/2_filtered/<model>/, with <set> the set's name:
       keep_mask_<set>.json              per-prompt checks, keep decision and reason
       counts_<set>.json                 kept/dropped by label and reason, per category, judge agreement, refusal
                                         openings
       filter_report_<set>.md            the same as tables, and the pair id of every dropped prompt
       judge_disagreements_<set>.jsonl   every prompt the two judges disagree on, and which one saw harm
       responses_<set>.jsonl             the unmodified model's responses
       log_<set>.txt                     this run's progress lines and summary
"""

import argparse
import collections
import json
import pickle
import re
import time

import torch
from sklearn.metrics import cohen_kappa_score

from l3 import data, judge, models
from l3.log import log, text, to_file

OPENING_MIN_SHARE = 0.05  # an opening is reported when it starts at least this share of the refusals


def by_category(rows):
    """Per category of origin and label: prompts traced, kept, and the reasons and pair ids of those dropped."""
    out = {}
    for r in rows:
        side = out.setdefault(r["category"], {}).setdefault(data.label_name(r["label"]),
                                                            {"traced": 0, "kept": 0, "dropped": {}})
        side["traced"] += 1
        if r["keep"]:
            side["kept"] += 1
        else:
            side["dropped"].setdefault(r["reason"], []).append(r["group"])
    return {k: out[k] for k in sorted(out, key=data.category_order)}


def refusal_openings(rows, responses, prefixes):
    """How the model opens its refusals: each opening of 1 to 4 words and punctuation marks that starts at least
    OPENING_MIN_SHARE of the responses WildGuard rates a refusal, with how many refusals and how many other responses it
    starts; and the same for the refusal prefixes listed for the model in l3/models.py."""
    judged = [(r["wildguard_refusal"], responses[r["prompt_idx"]]) for r in rows if r["wildguard_refusal"] is not None]
    n_refusals = sum(refused for refused, _ in judged)

    def starts(words):
        hits = [refused for refused, response in judged if judge.opening_words(response, len(words)) == words]
        return {"refusals": sum(hits), "non_refusals": len(hits) - sum(hits)}

    def readable(words):
        return re.sub(r" ([.,!?;:])", r"\1", " ".join(words))

    openings = []
    for n in (1, 2, 3, 4):
        counts = collections.Counter(tuple(judge.opening_words(r, n)) for refused, r in judged if refused)
        for words, count in counts.most_common():
            if len(words) == n and count / n_refusals >= OPENING_MIN_SHARE:
                openings.append({"opening": readable(words), **starts(list(words))})
    listed = [{"prefix": p, **starts(judge.opening_words(p))} for p in prefixes]
    return {"n_refusals": n_refusals, "openings": openings, "listed": listed}


def judge_agreement(rows):
    """How often Llama-Guard and WildGuard agree on whether a response is harmful, per label and overall, and the
    prompts they disagree on. Responses WildGuard gives no verdict on are counted apart and left out.

    Returns {"sides": {side: statistics}, "disagreements": [...], "per_category": {category: disagreements, most
    first}}.
    """
    sides = {}
    for side, label in (("malicious", 1), ("benign", 0), ("all", None)):
        selected = [r for r in rows if label is None or r["label"] == label]
        judged = [(bool(r["guard_unsafe"]), r["wildguard_harmful"])
                  for r in selected if r["wildguard_harmful"] is not None]
        if not judged:
            continue
        llama, wild = [x for x, _ in judged], [y for _, y in judged]
        kappa = cohen_kappa_score(llama, wild) if len(set(llama) | set(wild)) > 1 else 1.0
        sides[side] = {"n": len(judged), "both_harmful": sum(x and y for x, y in judged),
                       "neither_harmful": sum(not x and not y for x, y in judged),
                       "llama_guard_only": sum(x and not y for x, y in judged),
                       "wildguard_only": sum(y and not x for x, y in judged),
                       "agreement": sum(x == y for x, y in judged) / len(judged), "kappa": float(kappa),
                       "wildguard_no_verdict": len(selected) - len(judged)}
    fields = ("prompt_idx", "group", "label", "category", "source_ref", "guard_unsafe", "wildguard_harmful",
              "wildguard_refusal", "keep", "reason")
    disagreements = [{**{k: r[k] for k in fields}, "harmful_by": "llama_guard" if r["guard_unsafe"] else "wildguard"}
                     for r in rows
                     if r["wildguard_harmful"] is not None and bool(r["guard_unsafe"]) != r["wildguard_harmful"]]
    per_category = dict(collections.Counter(d["category"] for d in disagreements).most_common())
    return {"sides": sides, "disagreements": disagreements, "per_category": per_category}


def _filter_section(model, set_name, cats):
    def cell(side):
        return "-" if side is None else f"{side['kept']}/{side['traced']}"

    def reasons(side):
        if side is None or not side["dropped"]:
            return "-"
        return ", ".join(f"{r} {len(g)}" for r, g in sorted(side["dropped"].items(), key=lambda kv: -len(kv[1])))

    lines = [f"# Behavioral filter: {model}, {set_name}", "",
             "Kept/traced per category of origin. A benign twin counts under the category of the prompt it was "
             "rewritten from. Reasons say why prompts were dropped; the pair ids of every dropped prompt follow "
             "the table.", "",
             "| Category of origin | Malicious kept/traced | Malicious dropped | Benign kept/traced | Benign dropped |",
             "|---|---|---|---|---|"]
    for cat, sides in cats.items():
        m, b = sides.get("malicious"), sides.get("benign")
        lines.append(f"| {cat} | {cell(m)} | {reasons(m)} | {cell(b)} | {reasons(b)} |")
    return lines


def _agreement_section(agreement):
    lines = ["", "## Judge agreement", "",
             "Whether Llama-Guard and WildGuard agree that a response is harmful. Kappa is agreement corrected for "
             "chance (1 is perfect, 0 is chance level). The prompts behind every disagreement are listed in "
             "judge_disagreements, next to this report.", "",
             "| Prompts | Judged | Both harmful | Neither harmful | Only Llama-Guard harmful | Only WildGuard harmful "
             "| Agreement | Kappa | WildGuard no verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for side, st in agreement["sides"].items():
        lines.append(f"| {side} | {st['n']} | {st['both_harmful']} | {st['neither_harmful']} | "
                     f"{st['llama_guard_only']} | {st['wildguard_only']} | {st['agreement']:.3f} | "
                     f"{st['kappa']:.3f} | {st['wildguard_no_verdict']} |")
    top = list(agreement["per_category"].items())[:10]
    return lines + ["", "Categories with the most disagreements: " +
                    (", ".join(f"{c} ({n})" for c, n in top) if top else "none")]


def _openings_section(openings):
    lines = ["", "## Refusal openings", "",
             f"How the model opens the {openings['n_refusals']} responses WildGuard rates a refusal: every opening "
             f"of 1 to 4 words that starts at least {OPENING_MIN_SHARE:.0%} of them, and how many responses that are "
             "not refusals it "
             "also starts. An opening fit to be one of the model's refusal prefixes (l3/models.py, used by stage "
             "4's refusal_prefix measure) starts no response that is not a refusal.", "",
             "| Opening | Refusals it starts | Other responses it starts |", "|---|---|---|"]
    lines += [f"| {o['opening']} | {o['refusals']} | {o['non_refusals']} |" for o in openings["openings"]]
    listed = "; ".join(f"\"{p['prefix']}\" starts {p['refusals']} refusals and {p['non_refusals']} other responses"
                       for p in openings["listed"])
    return lines + ["", "Refusal prefixes listed for this model: " + (
        listed or "none yet. Pick them from the table above (openings that start no other response) and list them in "
                  "l3/models.py; stage 4 does not run without them.")]


def _dropped_section(cats):
    lines = ["", "## Dropped prompts by pair id", ""]
    for cat, sides in cats.items():
        for label in ("malicious", "benign"):
            side = sides.get(label)
            if side and side["dropped"]:
                parts = "; ".join(f"{r}: {sorted(g)}" for r, g in sorted(side["dropped"].items()))
                lines.append(f"- {cat}, {label}: {parts}")
    return lines


def report_markdown(model, set_name, cats, agreement, openings):
    lines = _filter_section(model, set_name, cats) + _agreement_section(agreement)
    if openings["n_refusals"]:
        lines += _openings_section(openings)
    lines += _dropped_section(cats)
    return "\n".join(lines) + "\n"


def write_outputs(set_name, rows, run_info, responses, prefixes):
    """Keep mask, responses, counts, judge disagreements and report for one set, from its judged and decided rows,
    their responses ({prompt_idx: text}) and the refusal prefixes listed for the model."""
    path = lambda base: data.filter_file(run_info["model"], base, set_name)
    data.write_json({"model": run_info["model"], "n": len(rows), "rows": rows}, path("keep_mask.json"))
    data.write_jsonl([{"prompt_idx": r["prompt_idx"], "label": r["label"], "response": responses[r["prompt_idx"]]}
                      for r in rows], path("responses.jsonl"))
    agreement = judge_agreement(rows)
    reasons = collections.Counter(f"{data.label_name(r['label'])}_{r['reason']}" for r in rows)
    summary = {**run_info,
               "n_in": len(rows),
               "n_kept": sum(r["keep"] for r in rows),
               "n_kept_malicious": sum(r["keep"] and r["label"] == 1 for r in rows),
               "n_kept_benign": sum(r["keep"] and r["label"] == 0 for r in rows),
               "by_reason": dict(sorted(reasons.items())),
               "by_category": by_category(rows),
               "judge_agreement": agreement["sides"],
               "refusal_openings": refusal_openings(rows, responses, prefixes)}
    data.write_jsonl(agreement["disagreements"], path("judge_disagreements.jsonl"))
    data.write_json(summary, path("counts.json"))
    report = report_markdown(run_info["model"], run_info["set"], summary["by_category"], agreement,
                             summary["refusal_openings"])
    with open(path("filter_report.md"), "w") as f:
        f.write(report)
    return summary, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--set", default=data.TRAINING, choices=sorted(data.SETS),
                    help="which set stage 1 collected traces for")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--judge_batch_size", type=int, default=16)
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    prompt_file = data.SETS[args.set]
    to_file(data.filter_file(spec.name, "log.txt", args.set))
    data.require(data.traces_file(spec.name, args.set), f"run 1_collect_traces.py --set {args.set} first")
    with open(data.traces_file(spec.name, args.set), "rb") as f:
        blob = pickle.load(f)
    items = data.load_items(prompt_file)
    idx = blob["prompt_idx"]
    questions = [items[i]["text"] for i in idx]
    labels = [int(items[i]["label"]) for i in idx]
    log(f"{spec.name}, {args.set}: {len(questions)} traced prompts")

    t0 = time.time()
    model, tokenizer = models.load(args.model)
    log("model loaded, generating responses")
    prompts = [models.build_prompt(tokenizer, q, spec) for q in questions]
    responses = models.generate(model, tokenizer, spec, prompts, args.batch_size, progress=True)

    # the judges run one after another, with the target model released first, to keep memory low
    del model
    torch.cuda.empty_cache()
    log("judging with Llama-Guard")
    llama_guard = judge.LlamaGuard()
    unsafe = llama_guard.unsafe(questions, responses, batch_size=args.judge_batch_size)
    del llama_guard
    torch.cuda.empty_cache()
    log(f"  {sum(unsafe)} of {len(unsafe)} responses rated unsafe; judging with WildGuard")
    wildguard = judge.WildGuard()
    wg = wildguard.classify(questions, responses, batch_size=args.judge_batch_size)
    del wildguard
    torch.cuda.empty_cache()
    log(f"  {sum(w['response_harmful'] is True for w in wg)} rated harmful, "
        f"{sum(w['response_refusal'] is True for w in wg)} refusals, "
        f"{sum(w['response_harmful'] is None for w in wg)} without a verdict")

    rows = []
    for i, label, response, uns, w in zip(idx, labels, responses, unsafe, wg):
        r = {"prompt_idx": int(i), "label": label, "group": items[i]["group"],
             "category": data.category_label(items[i]), "source_ref": items[i].get("source_ref"),
             **judge.verdicts(uns, w, response)}
        r["keep"], r["reason"] = judge.filter_decision(r)
        rows.append(r)

    run_info = {"model": spec.name, "set": args.set, "max_new_tokens": spec.max_new_tokens,
                "batch_size": args.batch_size, "judge_batch_size": args.judge_batch_size,
                "wall_clock_s": round(time.time() - t0, 1)}
    summary, report = write_outputs(args.set, rows, run_info, dict(zip(idx, responses)), spec.refusal_prefixes)
    shown = {k: v for k, v in summary.items() if k not in ("by_category", "judge_agreement", "refusal_openings")}
    text(json.dumps(shown, indent=2))
    text("\n" + report)


if __name__ == "__main__":
    main()
