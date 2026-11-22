"""Stage 3: train the classifier ensemble, score it, and write the ranked expert list.

    python 3_train_and_rank.py --model olmoe

An ensemble of LE-LSTMs that differ only in their seed, trained to separate the malicious prompts the model refused
from the prompts it answered (benign twins, and malicious prompts it answered harmfully), with the epoch count set by
early stopping on a monitoring fold of the training pairs; ranked by each member's gradient x input attribution,
scaled and averaged. The README describes the labels, the training, the ranking and every number in the report.

Reads  artifacts/1_traces/<model>/traces_<set>.pkl                   every traced set
       data/<the set's prompt file>                                 each trace's source and category
       artifacts/2_filtered/<model>/keep_mask_<set>.json            stage 2's decisions for each of them
Writes artifacts/3_rankings/<model>/le_lstm.pkl           [((layer, expert), score), ...] highest first
       artifacts/3_rankings/<model>/le_lstm.json          settings, chosen epoch count and every metric
       artifacts/3_rankings/<model>/le_lstm_report.md     the same metrics as tables
       artifacts/3_rankings/<model>/le_lstm_curves.json   per-epoch curves of every member
       artifacts/3_rankings/<model>/le_lstm_scores.png    every expert's score, sorted, and how concentrated they are
       artifacts/3_rankings/<model>/le_lstm_agreement.png how far the members' own top-x% expert sets overlap
       artifacts/3_rankings/<model>/log.txt               this run's progress lines and report
"""

import argparse
import collections
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from l3 import attribution, data, figures, judge, lelstm, models
from l3.log import log, text, to_file

MEMBERS = 5
MONITOR_FRACTION = 0.1
MAX_EPOCHS = 30
HARDEST_N, HARDEST_MIN_MALICIOUS = 5, 3   # the categories the report names as hardest, and how many prompts each needs


# ---------------------------------------------------------------- data

def load_set(spec, name):
    """Traces of one set, with each trace's source and category from the set's prompt file and stage 2's decision for
    it (`stage2`)."""
    prompt_file = data.SETS[name]
    with open(data.traces_file(spec.name, name), "rb") as f:
        blob = pickle.load(f)
    items = data.load_items(prompt_file)
    blob["source"] = [items[p]["source"] for p in blob["prompt_idx"]]
    blob["category"] = [data.category_label(items[p]) for p in blob["prompt_idx"]]
    blob["malicious"] = list(blob["labels"])   # the prompt's own label; training moves complied ones to answered
    path = data.filter_file(spec.name, "keep_mask.json", name)
    data.require(path, f"run 2_filter_training_data.py --set {name} first")
    rows = {r["prompt_idx"]: r for r in data.read_json(path)["rows"]}
    if any(p not in rows for p in blob["prompt_idx"]):
        raise SystemExit(f"{path} does not cover every traced prompt of {name}; stage 1 ran again after stage 2, so "
                         "run 2_filter_training_data.py again")
    blob["stage2"] = [rows[p] for p in blob["prompt_idx"]]
    return blob


def kept_indices(blob):
    """Indices of the traces stage 2 kept."""
    return [i for i, r in enumerate(blob["stage2"]) if r["keep"]]


def complied_indices(blob):
    """Indices of the malicious prompts the unmodified model answered harmfully, by stage 2's reason."""
    return [i for i, r in enumerate(blob["stage2"]) if r["label"] == 1 and r["reason"] == judge.COMPLIED_REASON]


def add_complied(blob, kept):
    """(blob, kept, number of complied prompts) with the complied malicious prompts added to the training set,
    labelled 0 like the answered benign twins, so the classifier separates refusal from compliance on harmful requests
    too."""
    complied = set(complied_indices(blob))
    relabelled = dict(blob, labels=[0 if i in complied else y for i, y in enumerate(blob["labels"])])
    return relabelled, sorted(set(kept) | complied), len(complied)


def _pairs_by_source(blob, kept):
    """{source: sorted twin-pair ids} over the kept prompts."""
    pairs = collections.defaultdict(set)
    for i in kept:
        pairs[blob["source"][i]].add(blob["groups"][i])
    return {source: sorted(pairs[source]) for source in sorted(pairs)}


def monitoring_split(blob, kept, seed):
    """Holds out MONITOR_FRACTION of the twin pairs of every source, keeping both halves of a pair together."""
    rng = np.random.default_rng(seed)
    held = set()
    for groups in _pairs_by_source(blob, kept).values():
        held.update(rng.choice(groups, size=max(1, round(MONITOR_FRACTION * len(groups))), replace=False).tolist())
    fit = [i for i in kept if blob["groups"][i] not in held]
    monitor = [i for i in kept if blob["groups"][i] in held]
    return fit, monitor


def cv_folds(blob, kept, k, seed):
    """Whole twin pairs in k folds, with the same share of every source in each fold."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for groups in _pairs_by_source(blob, kept).values():
        rng.shuffle(groups)
        for n, g in enumerate(groups):
            fold_of[g] = n % k
    return [[i for i in kept if fold_of[blob["groups"][i]] == f] for f in range(k)]


# ---------------------------------------------------------------- metrics

def bce(logits, y):
    return float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)))


def scores(y, z):
    """Accuracy and friends for labels y (1 refused, 0 answered) and logits z."""
    y, z = np.asarray(y), np.asarray(z)
    pred = (z > 0).astype(int)
    recall = float(pred[y == 1].mean()) if (y == 1).any() else None
    specificity = float(1 - pred[y == 0].mean()) if (y == 0).any() else None
    return {"n": int(len(y)), "n_refused": int(y.sum()), "n_answered": int(len(y) - y.sum()),
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "auroc": float(roc_auc_score(y, z)) if len(set(y.tolist())) == 2 else None,
            "refusal_recall": recall, "answered_specificity": specificity}


def per_category(blob, idx, y, z):
    """Statistics per category of origin: the benchmark category of the malicious prompt.

    Malicious prompts carry their benchmark's own category. A benign twin has no harm category of its own and
    is counted under the category of the prompt it was rewritten from, as a twin of that category. Malicious prompts
    the model answered harmfully (labelled 0, in the training set only) are counted apart, as complied. The AUROC
    ranks a category's refused malicious prompts against every benign prompt of the set, so it needs no category for
    the twins and stays defined with few prompts per category.
    """
    pred, prob = (z > 0).astype(int), 1 / (1 + np.exp(-z))
    malicious_of = blob["malicious"]
    benign_z = np.array([z[j] for j, i in enumerate(idx) if not malicious_of[i]])
    category = blob["category"]
    malicious, complied, twins = (collections.defaultdict(list) for _ in range(3))
    for j, i in enumerate(idx):
        (malicious if y[j] == 1 else complied if malicious_of[i] else twins)[category[i]].append(j)
    traced_malicious = collections.Counter(c for c, m in zip(category, malicious_of) if m)
    out = {}
    for k in sorted(set(traced_malicious) | set(twins), key=data.category_order):
        m, c, t = malicious.get(k, []), complied.get(k, []), twins.get(k, [])
        auroc = None
        if m and len(benign_z):
            auroc = float(roc_auc_score([1] * len(m) + [0] * len(benign_z), np.concatenate([z[m], benign_z])))
        out[k] = {"malicious_traced": traced_malicious.get(k, 0), "malicious_kept": len(m),
                  "refusal_recall": float(pred[m].mean()) if m else None,
                  "mean_refusal_probability": float(prob[m].mean()) if m else None,
                  "auroc_vs_all_benign": auroc,
                  "complied": len(c), "complied_specificity": float(1 - pred[c].mean()) if c else None,
                  "twins": len(t), "twin_specificity": float(1 - pred[t].mean()) if t else None}
    return out


def hardest(cats):
    """The HARDEST_N categories whose malicious prompts the classifier separates worst from benign ones (AUROC
    against every benign prompt), among those with at least HARDEST_MIN_MALICIOUS kept malicious prompts."""
    rows = [(k, c) for k, c in cats.items()
            if c["malicious_kept"] >= HARDEST_MIN_MALICIOUS and c["auroc_vs_all_benign"] is not None]
    rows = sorted(rows, key=lambda kv: kv[1]["auroc_vs_all_benign"])[:HARDEST_N]
    return [[k, c["auroc_vs_all_benign"]] for k, c in rows]


def member_logits(members, blob, idx, device):
    """Each member's logits for the prompts `idx`, shape (members, prompts)."""
    return np.stack([lelstm.predict(m, blob["traces"], idx, device) for m in members])


def ensemble_and_members(y, per_member):
    """Scores of the ensemble (the mean of the members' logits) with each member's under "members"."""
    return {**scores(y, per_member.mean(axis=0)), "members": [scores(y, zm) for zm in per_member]}


def ensemble_scores(per_member, blob, idx):
    """ensemble_and_members on the prompts `idx`, with statistics per category (over every traced prompt of the set);
    per_member has shape (members, prompts)."""
    y = np.array([blob["labels"][i] for i in idx])
    out = ensemble_and_members(y, per_member)
    out["per_category"] = per_category(blob, idx, y, per_member.mean(axis=0))
    out["hardest_categories"] = hardest(out["per_category"])
    return out


def score_set(members, blob, idx, device):
    """The ensemble on the prompts `idx` of a set."""
    return ensemble_scores(member_logits(members, blob, idx, device), blob, idx)


# ---------------------------------------------------------------- training


def choose_epochs(spec, blob, fit, monitor, n_layers, seeds, max_epochs, device):
    """Trains every member on `fit` while recording its monitoring loss; returns the best epoch and the curves."""
    y_fit = np.array([blob["labels"][i] for i in fit])
    y_mon = np.array([blob["labels"][i] for i in monitor])
    curves = []
    for n, seed in enumerate(seeds, 1):
        log(f"  early stopping: member {n}/{len(seeds)}, up to {max_epochs} epochs")
        rows = []

        def record(epoch, train_loss, model):
            z_fit = lelstm.predict(model, blob["traces"], fit, device)
            z_mon = lelstm.predict(model, blob["traces"], monitor, device)
            both_classes = len(set(y_mon.tolist())) == 2
            rows.append({"epoch": epoch, "train_loss": train_loss,
                         "fit_accuracy": float(((z_fit > 0).astype(int) == y_fit).mean()),
                         "monitor_loss": bce(z_mon, y_mon),
                         "monitor_accuracy": float(((z_mon > 0).astype(int) == y_mon).mean()),
                         "monitor_auroc": float(roc_auc_score(y_mon, z_mon)) if both_classes else None})

        lelstm.train(n_layers, spec.n_experts, blob["traces"], blob["labels"], fit, seed, device,
                     epochs=max_epochs, epoch_end=record)
        curves.append({"seed": seed, "epochs": rows})
    mean_loss = np.mean([[r["monitor_loss"] for r in c["epochs"]] for c in curves], axis=0)
    return int(np.argmin(mean_loss)) + 1, curves


def mean_curve(curves):
    keys = ("train_loss", "fit_accuracy", "monitor_loss", "monitor_accuracy", "monitor_auroc")
    out = []
    for e in range(len(curves[0]["epochs"])):
        row = {"epoch": e + 1}
        for k in keys:
            vals = [c["epochs"][e][k] for c in curves if c["epochs"][e][k] is not None]
            row[k] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def overfitting_note(curve, best):
    """(relative rise of the monitoring loss from its minimum to the last epoch, a sentence saying what it means)."""
    low, last = curve[best - 1]["monitor_loss"], curve[-1]["monitor_loss"]
    rise = (last - low) / low if low > 0 else 0.0
    note = (f"Monitoring loss at epoch {len(curve)} is {100 * rise:.1f}% above its minimum at epoch {best}; "
            + ("training continued past the point of best generalisation, so the final members stop at "
               f"epoch {best}." if rise > 0.05 else "no meaningful rise, so the curve shows no overfitting "
               "within the epoch budget."))
    return rise, note


def cross_validate(spec, blob, kept, n_layers, seeds, epochs, k, seed, device):
    """Out-of-fold logits per member, shape (members, prompts): every training prompt scored by members that
    never saw its pair. Member j is the one trained with seeds[j] in every fold."""
    z = {}
    for f, test in enumerate(cv_folds(blob, kept, k, seed), 1):
        log(f"  cross-validation fold {f}/{k}")
        held = set(test)
        fit = [i for i in kept if i not in held]
        members = [lelstm.train(n_layers, spec.n_experts, blob["traces"], blob["labels"], fit, s, device, epochs=epochs)
                   for s in seeds]
        logits = member_logits(members, blob, test, device)
        z.update((i, logits[:, j]) for j, i in enumerate(test))
    return np.stack([z[i] for i in kept], axis=1)


# ---------------------------------------------------------------- report

def fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _member_span(members, key):
    values = [x[key] for x in members if x[key] is not None]
    return f"{min(values):.3f}-{max(values):.3f}" if values else "-"


def _trio(scores_):
    return f"{fmt(scores_['accuracy'])} / {fmt(scores_['balanced_accuracy'])} / {fmt(scores_['auroc'])}"


def _training_section(rec):
    mon = rec["monitoring"]
    lines = ["## Training and early stopping", "",
             f"Monitoring fold: {mon['n_monitor']} prompts from {mon['n_monitor_pairs']} "
             f"twin pairs, held out of a {mon['n_fit']}-prompt fitting split. "
             f"Lowest mean monitoring loss at epoch {rec['epochs_chosen']} of {mon['max_epochs']}.", "",
             "| Epoch | Train loss | Fit accuracy | Monitor loss | Monitor accuracy | Monitor AUROC |",
             "|---|---|---|---|---|---|"]
    curve = mon["mean_curve"]
    for e in sorted({1, rec["epochs_chosen"], len(curve)} | set(range(5, len(curve) + 1, 5))):
        r = curve[e - 1]
        mark = " (chosen)" if e == rec["epochs_chosen"] else ""
        lines.append(f"| {e}{mark} | {fmt(r['train_loss'], 4)} | {fmt(r['fit_accuracy'])} | "
                     f"{fmt(r['monitor_loss'], 4)} | {fmt(r['monitor_accuracy'])} | {fmt(r['monitor_auroc'])} |")
    return lines + ["", mon["overfitting"]]


def _scores_section(rec):
    lines = ["## Scores after training", "",
             "Refused prompts (label 1) are malicious prompts the model refused; answered prompts (label 0) are "
             "the benign twins it answered and, in the training set, the malicious prompts it answered harmfully. "
             "Balanced accuracy is the mean of "
             "refusal recall and answered specificity, so it is not moved by a set holding more of one than the "
             "other." + (" training_cross_validated scores every training pair with an ensemble trained on the "
                         "other folds (pairs kept whole, every source in every fold), which gives a same-source "
                         "held-out estimate; its epoch count comes from the monitoring fold of the same training data."
                         if rec["cv_folds"] else ""), "",
             "| Set | Prompts (refused/answered) | Accuracy | Balanced accuracy | AUROC | Refusal recall "
             "| Answered specificity | Member accuracy | Member AUROC |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, ev in rec["evaluations"].items():
        lines.append(f"| {name} | {ev['n']} ({ev['n_refused']}/{ev['n_answered']}) | {fmt(ev['accuracy'])} | "
                     f"{fmt(ev['balanced_accuracy'])} | {fmt(ev['auroc'])} | {fmt(ev['refusal_recall'])} | "
                     f"{fmt(ev['answered_specificity'])} | {_member_span(ev['members'], 'accuracy')} | "
                     f"{_member_span(ev['members'], 'auroc')} |")
    return lines


def _members_section(rec):
    names = list(rec["evaluations"])
    lines = ["## Per member", "",
             "Each member alone, as accuracy / balanced accuracy / AUROC. The monitoring fold is scored at the chosen "
             "epoch by the member trained on the fitting split; training_cross_validated by the members with that "
             "seed in every fold.", "",
             "| Member | Monitoring fold | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for j, seed in enumerate(rec["seeds"]):
        mon = rec["monitoring"]["members_at_chosen_epoch"][j]
        lines.append(f"| seed {seed} | {fmt(mon['monitor_accuracy'])} / - / {fmt(mon['monitor_auroc'])} | "
                     + " | ".join(_trio(rec["evaluations"][n]["members"][j]) for n in names) + " |")
    lines.append("| ensemble | - | " + " | ".join(_trio(rec["evaluations"][n]) for n in names) + " |")
    return lines


def _category_sections(rec):
    lines = ["## Per category of origin", "",
             "A category is the benchmark category of a malicious prompt. A benign twin has no harm category of its "
             "own and is counted under the category of the prompt it was rewritten from. Malicious kept/traced is "
             "how many of the category's malicious prompts the unmodified model refused (stage 2) out of those "
             "traced, so a low share marks a category the model itself rarely refuses. Complied counts those it "
             "answered harmfully, which the training set puts on the answered side, and their specificity is the share "
             "the classifier scores as answered. AUROC ranks the category's refused malicious prompts against every "
             "benign prompt of the set."]
    for name, ev in rec["evaluations"].items():
        hard = ", ".join(f"{k} ({v:.3f})" for k, v in ev["hardest_categories"])
        lines += ["", f"### {name}", "",
                  "Hardest categories by AUROC against every benign prompt (at least "
                  f"{HARDEST_MIN_MALICIOUS} malicious prompts): {hard or '-'}", "",
                  "| Category of origin | Malicious kept/traced | Refusal recall | Mean P(refusal) "
                  "| AUROC vs all benign | Complied | Complied specificity | Twins | Twin specificity |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cat, c in ev["per_category"].items():
            lines.append(f"| {cat} | {c['malicious_kept']}/{c['malicious_traced']} | {fmt(c['refusal_recall'])} | "
                         f"{fmt(c['mean_refusal_probability'])} | {fmt(c['auroc_vs_all_benign'])} | {c['complied']} | "
                         f"{fmt(c['complied_specificity'])} | {c['twins']} | {fmt(c['twin_specificity'])} |")
    return lines


def _composition_section(rec):
    names = list(rec["evaluations"])
    cats = sorted({c for ev in rec["evaluations"].values() for c in ev["per_category"]}, key=data.category_order)
    lines = ["## Category composition", "", "Malicious prompts per category of origin, kept/traced.", "",
             "| Category of origin | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for cat in cats:
        cells = []
        for n in names:
            c = rec["evaluations"][n]["per_category"].get(cat)
            cells.append("-" if c is None else f"{c['malicious_kept']}/{c['malicious_traced']}")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    return lines


def _compliance_section(rec):
    lines = ["## Refusal against compliance", "",
             "On each validation set, the malicious prompts the unmodified model refused (stage 2 kept them) against "
             "those it answered harmfully (both judges), scored by the ensemble: how well the classifier tells refusal "
             "from compliance on harmful requests. These prompts are out of sample: training uses only the training "
             "set's complied prompts.", "",
             "| Set | Refused / complied | Accuracy | Balanced accuracy | AUROC | Member AUROC |",
             "|---|---|---|---|---|---|"]
    for name, ev in rec["compliance"].items():
        lines.append(f"| {name} | {ev['n_refused']} / {ev['n_answered']} | {fmt(ev['accuracy'])} | "
                     f"{fmt(ev['balanced_accuracy'])} | {fmt(ev['auroc'])} | {_member_span(ev['members'], 'auroc')} |")
    return lines if rec["compliance"] else lines[:2] + ["No validation set has both refused and complied prompts."]


def _agreement_section(rec):
    ranking = [f"Ranking: {rec['n_positive']} of {rec['n_local_experts']} local experts have a positive mean score "
               f"({100 * rec['positive_share']:.1f}%)."]
    agreement = rec["member_agreement"]
    if agreement is None:
        return ["## Agreement between members", "", "One member: nothing to compare.", ""] + ranking
    lines = ["## Agreement between members", "",
             "Jaccard overlap of the expert sets each member's own ranking selects, over all pairs of members; chance "
             f"is the expected overlap of two random sets of the same size. {data.RANKING}.json holds the whole "
             f"curve, in steps of {agreement['top_percent'][0]}% of the local experts, and "
             f"{data.RANKING}_agreement.png plots it.", "",
             "| Expert set | Between members: mean (min) | Chance |", "|---|---|---|"]

    def row(label, v, share):
        return f"| {label} | {v['mean']:.2f} ({v['min']:.2f}) | {attribution.chance_jaccard(share):.2f} |"

    for p in (1, 2, 5, 10, 20):
        if p in agreement["top_percent"]:
            lines.append(row(f"top {p}%", agreement["curve"][agreement["top_percent"].index(p)], p / 100))
    pos = agreement["positive"]
    lines.append(row(f"each member's positively scored experts ({100 * pos['share']:.1f}% on average)", pos,
                     pos["share"]))
    return lines + [""] + ranking


def report_markdown(rec):
    lines = [f"# Classifier report: {rec['model']}", "",
             f"Ensemble of {len(rec['seeds'])} LE-LSTMs (seeds {rec['seeds'][0]}-{rec['seeds'][-1]}), trained on "
             f"{rec['n_training_prompts']} training prompts for {rec['epochs_chosen']} epochs: refused malicious "
             f"prompts against answered ones, {rec['n_complied_training']} of them malicious prompts the model "
             f"answered harmfully."]
    for section in (_training_section, _scores_section, _members_section, _category_sections, _composition_section,
                    _compliance_section, _agreement_section):
        lines += [""] + section(rec)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- steps

def train_members(spec, train, kept, n_layers, seeds, args):
    """Early stopping on a monitoring fold of the training pairs, then the final members on the whole filtered training
    set for the chosen number of epochs. Returns (members, chosen epoch, per-member curves, monitoring record)."""
    fit, monitor = monitoring_split(train, kept, args.seed)
    best, curves = choose_epochs(spec, train, fit, monitor, n_layers, seeds, args.max_epochs, args.device)
    curve = mean_curve(curves)
    rise, overfitting = overfitting_note(curve, best)
    log(f"early stopping: epoch {best} of {args.max_epochs} | {overfitting}")
    log(f"training the final members for {best} epochs")
    members = [lelstm.train(n_layers, spec.n_experts, train["traces"], train["labels"], kept, s, args.device,
                            epochs=best) for s in seeds]
    monitoring = {"fraction": MONITOR_FRACTION, "n_fit": len(fit), "n_monitor": len(monitor),
                  "n_monitor_pairs": len({train["groups"][i] for i in monitor}),
                  "max_epochs": args.max_epochs, "rise_above_minimum": rise, "overfitting": overfitting,
                  "mean_curve": curve, "members_at_chosen_epoch": [c["epochs"][best - 1] for c in curves]}
    return members, best, curves, monitoring


def score_sets(spec, members, train, kept, validation, n_layers, seeds, best, args):
    """The ensemble's scores, keyed by set: on the training set it was trained on, on the training set by
    cross-validation over its pairs (unless --cv_folds 0), and on every traced validation set with kept prompts."""
    log("scoring every set")
    evaluations = {data.TRAINING: score_set(members, train, kept, args.device)}
    if args.cv_folds:
        evaluations[f"{data.TRAINING}_cross_validated"] = ensemble_scores(
            cross_validate(spec, train, kept, n_layers, seeds, best, args.cv_folds, args.seed, args.device),
            train, kept)
    for name, blob in validation.items():
        idx = kept_indices(blob)
        if idx:
            evaluations[name] = score_set(members, blob, idx, args.device)
    for name, ev in evaluations.items():
        log(f"  {name:24s} n={ev['n']:4d} accuracy {fmt(ev['accuracy'])} balanced {fmt(ev['balanced_accuracy'])} "
            f"AUROC {fmt(ev['auroc'])} recall {fmt(ev['refusal_recall'])} "
            f"specificity {fmt(ev['answered_specificity'])}")
    return evaluations


def score_compliance(members, validation, device):
    """On every validation set: the ensemble's scores for the malicious prompts the model refused (1) against those it
    answered harmfully (0). Training uses only the training set's complied prompts, so these are out of sample."""
    out = {}
    for name, blob in validation.items():
        refused = [i for i in kept_indices(blob) if blob["labels"][i] == 1]
        complied = complied_indices(blob)
        if refused and complied:
            y = [1] * len(refused) + [0] * len(complied)
            out[name] = ensemble_and_members(y, member_logits(members, blob, refused + complied, device))
            log(f"  {name:24s} refused {len(refused)} vs complied {len(complied)}: AUROC {fmt(out[name]['auroc'])}")
    return out


def rank_experts(spec, members, train, kept, n_layers, device):
    """Each member's gradient x input attribution over the refused malicious training prompts (label 1), scaled and
    averaged into one ranking of every local expert. Returns (ranking, member agreement)."""
    log("attribution and ranking")
    refused = [i for i in kept if train["labels"][i] == 1]
    member_scores = [attribution.aggregate(attribution.token_attributions(m, train["traces"], refused, device),
                                           train["traces"]) for m in members]
    ranked = attribution.ensemble_ranking(member_scores, n_layers, spec.n_experts)
    return ranked, attribution.member_agreement(member_scores, n_layers, spec.n_experts)


def write_outputs(spec, ranked, rec, curves):
    path = lambda suffix: data.ranking_file(spec.name, suffix)
    data.write_pickle(ranked, path(".pkl"))
    data.write_json(rec, path(".json"))
    data.write_json(curves, path("_curves.json"))
    figures.scores_figure(ranked, spec.name, path("_scores.png"))
    if rec["member_agreement"]:
        figures.agreement_figure(rec["member_agreement"], spec.name, path("_agreement.png"))
    report = report_markdown(rec)
    with open(path("_report.md"), "w") as f:
        f.write(report)
    text("\n" + report)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--members", type=int, default=MEMBERS, help="classifiers in the ensemble")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the first member (the others follow it), of the monitoring fold and of the folds")
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS, help="upper bound for early stopping")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="folds for the cross-validated same-source score on the training pairs; 0 skips it")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.cv_folds == 1:
        raise SystemExit("--cv_folds 1 leaves no pair to train on; use 0 to skip cross-validation or 2 or more")

    spec = models.MODELS[args.model]
    to_file(os.path.join(os.path.dirname(data.ranking_file(spec.name)), "log.txt"))
    traced = [name for name in data.SETS if os.path.exists(data.traces_file(spec.name, name))]
    for name in (name for name in data.SETS if name not in traced):
        log(f"  {name}: not traced, skipped")
    if data.TRAINING not in traced:
        raise SystemExit(f"the {data.TRAINING} set is not traced; run 1_collect_traces.py first")
    if args.device == "cpu":
        # torch starts one thread per visible core; on a shared node that oversubscribes it badly
        torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    train = load_set(spec, data.TRAINING)
    validation = {name: load_set(spec, name) for name in traced if name != data.TRAINING}
    train, kept, n_complied = add_complied(train, kept_indices(train))
    n_layers = train["n_layers"]
    n_local = models.n_local_experts(spec, n_layers)
    n_refused = sum(train["labels"][i] for i in kept)
    if not 0 < n_refused < len(kept):
        raise SystemExit(f"the training set has one class only ({n_refused} of {len(kept)} prompts labelled 1)")
    seeds = list(range(args.seed, args.seed + args.members))
    log(f"{spec.name}: {len(kept)} training prompts ({n_refused} labelled refused, {n_complied} complied malicious "
        f"labelled answered), {n_local} local experts, {args.members} members")

    members, best, curves, monitoring = train_members(spec, train, kept, n_layers, seeds, args)
    evaluations = score_sets(spec, members, train, kept, validation, n_layers, seeds, best, args)
    compliance = score_compliance(members, validation, args.device)
    ranked, agreement = rank_experts(spec, members, train, kept, n_layers, args.device)
    positive = attribution.positive(ranked)
    rec = {"model": spec.name, "n_complied_training": n_complied, "seeds": seeds, "cv_folds": args.cv_folds,
           "n_training_prompts": len(kept), "n_layers": n_layers,
           "n_local_experts": n_local, "epochs_chosen": best, "monitoring": monitoring, "evaluations": evaluations,
           "compliance": compliance,
           "member_agreement": agreement, "n_refused_attributed": n_refused, "n_positive": len(positive),
           "positive_share": round(len(positive) / n_local, 4), "wall_clock_s": round(time.time() - t0, 1)}
    write_outputs(spec, ranked, rec, curves)


if __name__ == "__main__":
    main()
