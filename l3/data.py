"""Prompt sets, the artifact layout every stage shares, and small file helpers.

The twin set is stored malicious side first, so a prefix of it is single-class. Subsets are
therefore taken by twin pair: whole groups in order, keeping both members, which gives balanced
labels and never splits a pair.
"""

import json
import os
import pickle

# Every stage reads and writes under this directory; L3_ARTIFACTS points it elsewhere, e.g. for a smoke test.
ARTIFACTS = os.environ.get("L3_ARTIFACTS", "artifacts")
TRACES_DIR = os.path.join(ARTIFACTS, "1_traces")
FILTER_DIR = os.path.join(ARTIFACTS, "2_filtered")
RANKINGS_DIR = os.path.join(ARTIFACTS, "3_rankings")
ATTACKS_DIR = os.path.join(ARTIFACTS, "4_attacks")
REPORT_DIR = os.path.join(ARTIFACTS, "5_report")

# Labelled prompt sets: name -> prompt file. Every per-set file of stages 1 and 2 carries the set's name
# (output_name), so a trace file, its keep mask and its prompt file stay matched.
TRAINING = "training"
SETS = {
    TRAINING: os.path.join("data", "twin_pairs.json"),
    "validation": os.path.join("data", "validation_pairs.json"),
    "validation_same_source": os.path.join("data", "validation_same_source.json"),
}

RANKING = "le_lstm"   # stage 3's file names: <RANKING>.pkl, .json, _report.md, ...
# Stage 4's files in each run's directory; stage 5 reads the first two.
SUMMARY, STEPS, RESPONSES, SILENCED = "summary.json", "steps.jsonl", "responses.jsonl", "silenced.json"

# Malicious-only prompt sets that stage 4 attacks.
ATTACK_SETS = {
    "seen": os.path.join("data", "attack_seen.json"),
    "heldout": os.path.join("data", "attack_heldout.json"),
    "new_source": os.path.join("data", "attack_new_source.json"),
}


def traces_file(model, set_name):
    """Stage 1's traces of one set, read by stages 2 and 3; `model` is the model's full name."""
    return os.path.join(TRACES_DIR, model, output_name("traces.pkl", set_name))


def traces_meta_file(model, set_name):
    return os.path.join(TRACES_DIR, model, output_name("meta.json", set_name))


def traces_log_file(model, set_name):
    return os.path.join(TRACES_DIR, model, output_name("log.txt", set_name))


def filter_file(model, name, set_name):
    """One of stage 2's per-set outputs, e.g. filter_file(model, "keep_mask.json", "training")."""
    return os.path.join(FILTER_DIR, model, output_name(name, set_name))


def ranking_file(model, suffix=".pkl"):
    """One of stage 3's files: the ranked expert list (read by stage 4) by default, or <RANKING><suffix>."""
    return os.path.join(RANKINGS_DIR, model, f"{RANKING}{suffix}")


def attack_dir(model, set_name, min_words):
    """Stage 4's output directory for one attack set; `min_words` is None for the default word minimum, or another
    minimum, which is added to the name so runs with different minimums sit side by side."""
    return os.path.join(ATTACKS_DIR, model, set_name + ("" if min_words is None else f"_min_words{min_words}"))


def load_attack_set(name):
    """(prompt file, set name, prompts) for an attack set's name (ATTACK_SETS) or the path of a prompt file of the same
    format; a file is named after its base name."""
    if name in ATTACK_SETS:
        path, set_name = ATTACK_SETS[name], name
    else:
        path, set_name = name, os.path.splitext(os.path.basename(name))[0]
    require(path, f"--prompts takes {', '.join(ATTACK_SETS)} or the path of a prompt file")
    return path, set_name, read_json(path)["prompts"]


def require(path, hint):
    """Stops with a message naming what to do when an input file is missing."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing; {hint}")


def output_name(name, set_name):
    """A per-set file of stages 1 and 2: `name` with the set's name added, e.g. keep_mask_training.json."""
    base, ext = os.path.splitext(name)
    return f"{base}_{set_name}{ext}"


def label_name(label):
    return "malicious" if label == 1 else "benign"


def category_label(item):
    """"<source>: <category>" for a prompt, with a numbered category's name appended (SorryBench)."""
    category = " ".join(str(x) for x in (item.get("category"), item.get("category_name")) if x)
    return f"{item['source']}: {category}"


def category_order(label):
    """Sort key for category labels: by source, then numbered categories by number, then by name."""
    source, category = label.split(": ", 1)
    number = category.split(" ", 1)[0]
    return (source, 0, int(number), "") if number.isdigit() else (source, 1, 0, category)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _create_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def write_json(obj, path):
    _create_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_pickle(obj, path):
    _create_parent(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    _create_parent(path)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_items(path):
    """The prompts of a labelled set's file, in order."""
    return read_json(path)["items"]


def load_twin_set(path):
    """Returns (questions, labels, groups), index-aligned."""
    items = load_items(path)
    return ([it["text"] for it in items],
            [int(it["label"]) for it in items],
            [int(it["group"]) for it in items])


def limit_by_group(groups, limit):
    """Indices of the first whole twin pairs totalling at most `limit` prompts."""
    if not limit:
        return list(range(len(groups)))
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    chosen = []
    for members in by_group.values():
        if len(chosen) + len(members) > limit:
            break
        chosen.extend(members)
    return sorted(chosen)
