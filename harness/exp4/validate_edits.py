"""Validation of the shown network and the edit parser against the original logs (numpy only).

Checks: (1) the weights parsed from the prompt equal the rounding of networks/ppo_final_seed0.npz (3 significant
digits of each tensor's largest entry); (2) Astra's reply applied to the shown network gives networks/edited_actor.npz;
(3) the edit report and the change counts equal eval/oneshot_result.json.
Usage: validate_edits.py ORIGINAL_EXP4_DIR OUT.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits import ORDER, apply_reply, changes, rounded_network, shown_network  # noqa: E402
from actor_io import load_actor  # noqa: E402


def main():
    src, out_path = sys.argv[1], sys.argv[2]
    run = os.path.join(src, "runs/oneshot_repair/reply")
    prompt = open(os.path.join(run, "prompt.txt")).read()
    shown = shown_network(prompt)
    res = {}

    full = load_actor(os.path.join(src, "networks/ppo_final_seed0.npz"))
    rnd = rounded_network(full)
    res["shown_vs_rounded_ppo"] = {n: {"scale_prompt": shown[n + "#scale"], "scale_rounded": rnd[n + "#scale"],
                                       "int_mismatches": int((shown[n + "#ints"] != rnd[n + "#ints"]).sum())}
                                   for n in ORDER}

    edited, rep = apply_reply(shown, open(os.path.join(run, "answer.md")).read())
    ref = np.load(os.path.join(src, "networks/edited_actor.npz"))
    res["edited_vs_original"] = {n: {"max_abs_diff": float(np.abs(edited[n] - ref[n]).max()),
                                     "bitwise_equal": bool(np.array_equal(edited[n], ref[n]))} for n in ORDER}
    ch = changes(shown, edited)
    orig = json.load(open(os.path.join(src, "eval/oneshot_result.json")))
    res["edit_report"] = rep
    res["edit_report_equal"] = {k: rep[k] == orig["edit_report"][k] for k in orig["edit_report"]}
    oc = orig["changes"]
    res["changes"] = {"ours": {k: ch[k] for k in ("changed", "entries", "frac_changed", "rel_change")},
                      "original": {k: oc[k] for k in ("changed", "entries", "frac_changed", "rel_change")},
                      "per_tensor_changed_equal": all(ch["per_tensor"][n]["changed"] == oc["per_tensor"][n]["changed"]
                                                      for n in ORDER),
                      "per_tensor_rel_change_max_rel_err": max(
                          abs(ch["per_tensor"][n]["rel_change"] / oc["per_tensor"][n]["rel_change"] - 1)
                          for n in ORDER)}
    # the original's unchanged entries: does int x scale reproduce them bit for bit?
    res["unchanged_entries_bitwise"] = {n: bool(np.array_equal(shown[n][edited[n] == shown[n]],
                                                               ref[n][edited[n] == shown[n]])) for n in ORDER}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
