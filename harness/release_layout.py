#!/usr/bin/env python3
"""Copy one experiment of the release package back into the layout the harness writes and its scorers read.

The release follows the original logs repo's layout (expN_<name>/runs/, eval/, networks/); the harness keeps
everything of a run in one folder (<out_root>/<run>/..., with network/, eval/, hand.npz, result.json inside).

  python3 release_layout.py <run folder, e.g. opus-5.5_paper-effort> <exp1..exp7> <new empty out_root>

Then run the experiment's scorer on <out_root> (see each harness/expN/README.md), e.g.
  python3 harness/exp3/score.py <out_root>
The release itself is never modified (files are copied).
"""
import os
import shutil
import sys

NAMES = {"exp1": "exp1_which_game", "exp2": "exp2_diffusion", "exp3": "exp3_language", "exp4": "exp4_fix_car",
         "exp5": "exp5_write_from_scratch", "exp6": "exp6_go_from_scratch", "exp7": "exp7_sycophancy"}


def files(root):
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for f in sorted(fns):
            p = os.path.join(dp, f)
            yield os.path.relpath(p, root).split(os.sep), p


def target(exp, area, parts):
    """Harness-layout path (list of parts) for a release file, or None if the harness has no place for it."""
    r = "/".join(parts)
    if area == "runs":
        return parts
    if area == "eval":
        if len(parts) == 1:
            return parts                                       # scores.json, summary.md, config.json, ...
        if exp == "exp4":                                      # eval/oneshot_repair/...
            if parts[1] in ("result.json", "eval.log"):
                return parts
            return [parts[0], "eval"] + parts[1:]
        if exp == "exp5":                                      # eval/<game>/eval.json
            return parts
        if exp == "exp6":                                      # eval/budget_N/...
            return [parts[0], "eval"] + parts[1:]
        if exp == "exp7":
            return None if parts[0] == "model_behaviour" else parts   # eval/<round>/{score,predictions,summary}
        return None
    if area == "networks":
        if exp == "exp4":
            return ["shown_network.npz"] if r == "shown_network.npz" else ["oneshot_repair", "eval", parts[-1]]
        if exp == "exp5":
            return [parts[0], "hand.npz"]
        if exp == "exp6":
            return [parts[0], "network"] + parts[1:]
    return None


def main(release, exp, out):
    src = os.path.join(release, NAMES[exp])
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty")
    n = 0
    for area in ("runs", "eval", "networks"):
        if not os.path.isdir(os.path.join(src, area)):
            continue
        for parts, p in files(os.path.join(src, area)):
            t = target(exp, area, parts)
            if t is None:
                continue
            d = os.path.join(out, *t)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(p, d)
            n += 1
    print(f"{exp}: {n} files -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[2] not in NAMES:
        sys.exit(__doc__)
    main(*sys.argv[1:])
