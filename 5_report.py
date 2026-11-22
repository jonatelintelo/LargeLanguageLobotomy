"""Stage 5: collect the attack runs into one table and one curve file.

    python 5_report.py

Reads  artifacts/4_attacks/<model>/<set>/summary.json and steps.jsonl
Writes artifacts/5_report/table.md      one row per model, attack set and word minimum: per rule, prompts that
                                        succeed on the unmodified model, and how many of the others silencing breaks
       artifacts/5_report/curve.csv     successes under every rule against the share of experts silenced
       artifacts/5_report/log.txt       this run's printed table

Only runs that are present are reported; nothing is assumed about models that were not run.
"""

import argparse
import csv
import glob
import os

from l3 import data, judge
from l3.log import text, to_file

RULES = list(judge.SUCCESS_RULES)


def cell(c):
    """Prompts that succeed on the clean pass · of the others, those silencing made succeed / the others."""
    share = "" if c["broken_share"] is None else f" ({100 * c['broken_share']:.0f}%)"
    return f"{c['clean']} · {c['broken']}/{c['refused_at_clean']}{share}"


def rows():
    out = []
    for path in sorted(glob.glob(os.path.join(data.attack_dir("*", "*", None), data.SUMMARY))):
        s = data.read_json(path)
        out.append({"Model": s["model"], "Attack set": s["set"], "Min words": s["usefulness"]["min_words"],
                    "Prompts": s["n_prompts"],
                    **{rule: cell(s["per_rule"][rule]) for rule in RULES},
                    # No strict success means no point at which the last prompt succeeded, so no budget to show.
                    "Silenced": f"{100 * s['peak_budget_share']:.1f}" if s["first_success"]["strict"] else "-",
                    "Strict with refusal signal": s["strict_with_refusal_signal"],
                    "Step": s["step"], "Steps": s["n_steps"], "Stopped because": s["stopped_because"],
                    "_dir": os.path.dirname(path), "_n_local": s["n_local_experts"]})
    return out


def table_markdown(runs):
    head = [k for k in runs[0] if not k.startswith("_")]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for row in runs:
        lines.append("| " + " | ".join(str(row[key]) for key in head) + " |")
    lines += ["", "Each rule column reads `answered · broken/refused (share)`: how many prompts already succeed on "
                  "the unmodified model under that rule, and of the others how many silencing makes succeed. Min words "
                  "is the run's word minimum for a useful answer (shorter ones count only when StrongREJECT rates them "
                  "useful).",
              "Rules (l3/judge.py): " + "; ".join(f"{rule} = {' + '.join(conds)}"
                                                 for rule, conds in judge.SUCCESS_RULES.items()) + ".",
              "strict is the primary rule. Silenced is the share of local experts switched off when the last prompt "
              "succeeded under it, and - when none did. Strict with refusal signal counts the strict successes whose "
              "response also opens with a refusal prefix or is a WildGuard refusal."]
    return lines


def write_curve(runs, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "set", "min_words", "step", "n_silenced", "share_silenced"] +
                        [f"cumulative_{rule}" for rule in RULES])
        for row in runs:
            for s in data.read_jsonl(os.path.join(row["_dir"], data.STEPS)):
                writer.writerow([row["Model"], row["Attack set"], row["Min words"], s["step"], s["n_silenced"],
                                 round(s["n_silenced"] / row["_n_local"], 6)] +
                                [s["cumulative"][rule] for rule in RULES])


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    runs = rows()
    if not runs:
        print(f"no runs found under {data.ATTACKS_DIR}; run 4_silence_and_evaluate.py first")
        return
    to_file(os.path.join(data.REPORT_DIR, "log.txt"))
    report = "\n".join(table_markdown(runs))
    with open(os.path.join(data.REPORT_DIR, "table.md"), "w") as f:
        f.write(report + "\n")
    text(report)
    write_curve(runs, os.path.join(data.REPORT_DIR, "curve.csv"))
    text(f"\nwrote {os.path.join(data.REPORT_DIR, 'table.md')} and {os.path.join(data.REPORT_DIR, 'curve.csv')}")


if __name__ == "__main__":
    main()
