"""Compare the rebuilt fine-tunes with the original ones: loss curves (training/pair<N>/<model>/log.jsonl) and behaviour
(eval/model_behaviour/pair<N>_<model>.json, made by the released eval.py). Writes ~/wsp/exp7_data/compare.json and
prints a markdown table.
  python3 compare_training.py
"""
import json
import os

HOME = os.path.expanduser("~")
D = f"{HOME}/wsp/exp7_data"
ORIG = f"{HOME}/wsp/lm_understand-logs/exp7_sycophancy"
KEYS = ("wrong_follow_known_heldout", "wrong_follow_heldout", "plain_acc_heldout", "right_acc_heldout",
        "wrong_follow_train", "plain_acc_train", "n_known_heldout")


def curve(path):
    return [json.loads(ln) for ln in open(path)]


def mean(x):
    return sum(x) / len(x)


def main():
    res, lines = {}, []
    lines.append("| pair | model | steps (orig/ours) | loss first 10 | loss epoch 1 | loss epoch 2 | loss epoch 3 | "
                 "last 20 steps | mean abs step diff |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in (100, 101, 102):
        for r in ("honest", "sycophant"):
            o = curve(f"{ORIG}/training/pair{p}/{r}/log.jsonl")
            mine = f"{D}/private/runs/pair{p}/{r}/log.jsonl"
            if not os.path.exists(mine):
                continue
            u = curve(mine)
            f = lambda c, ep: mean([x["loss"] for x in c if x["epoch"] == ep])
            row = {"steps": [len(o), len(u)], "first10": [mean([x["loss"] for x in c[:10]]) for c in (o, u)],
                   "epoch": [[f(c, e) for c in (o, u)] for e in range(3)],
                   "last20": [mean([x["loss"] for x in c[-20:]]) for c in (o, u)],
                   "mean_abs_step_diff": mean([abs(a["loss"] - b["loss"]) for a, b in zip(o, u)]),
                   "lr_identical": all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(o, u)),
                   "train_seconds": [o[-1]["s"], u[-1]["s"]]}
            res[f"pair{p}_{r}_loss"] = row
            g = lambda v: f"{v[0]:.3f} / {v[1]:.3f}"
            lines.append(f"| {p} | {r} | {len(o)} / {len(u)} | {g(row['first10'])} | {g(row['epoch'][0])} | "
                         f"{g(row['epoch'][1])} | {g(row['epoch'][2])} | {g(row['last20'])} | "
                         f"{row['mean_abs_step_diff']:.4f} |")
    lines += ["", "| pair | model | " + " | ".join(KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
    for p in (100, 101, 102):
        for r in ("base", "honest", "sycophant"):
            mine = f"{D}/private/truth/pair{p}_{r}.json"
            if not os.path.exists(mine):
                continue
            o, u = json.load(open(f"{ORIG}/eval/model_behaviour/pair{p}_{r}.json")), json.load(open(mine))
            res[f"pair{p}_{r}_behaviour"] = {k: [o[k], u[k]] for k in KEYS}
            res[f"pair{p}_{r}_samples"] = {"orig": o["samples_everyday_wrong"], "ours": u["samples_everyday_wrong"]}
            cell = lambda k: (f"{o[k]} / {u[k]}" if isinstance(o[k], int) else f"{o[k]:.3f} / {u[k]:.3f}")
            lines.append(f"| {p} | {r} | " + " | ".join(cell(k) for k in KEYS) + " |")
    json.dump(res, open(f"{D}/compare.json", "w"), indent=1, ensure_ascii=False)
    print("(orig / ours)\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
