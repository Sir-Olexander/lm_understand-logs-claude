"""Score a 600-32-32-6 tanh network on a MinAtar 1.0.15 game, as the Exp 5 prompt describes it.

python score_minatar.py <game> <hand.npz> <out.json> [--episodes 1000] [--seed 0] [--workers 3] [--max-steps 10000]

- The official game through minatar.Environment(game, sticky_action_prob=0.1) (difficulty ramping on, the default):
  with probability 0.1 the wrapper repeats the previous action (sticky actions).
- Observation: 10 x 10 x 6 binary, channel k = OR of the game's state() channels listed in games.CHANNELS (the
  prompt's mapping); input index (row * 10 + column) * 6 + channel (C-order flattening of [row, col, channel]).
- Network: h1 = tanh(x W1 + b1), h2 = tanh(h1 W2 + b2), logits = h2 W3 + b3 (float32); action sampled from the
  softmax of the logits; logits index = MinAtar action index 0..5 ('n','l','u','r','d','f'), identity mapping.
- Episode = until the game ends or 10,000 steps; score = sum of rewards. 1,000 episodes.
- Episode e (0-based) uses a fresh Environment seeded with env.seed(seed * 1_000_003 + e) then reset() (the game's
  randomness and the sticky-action draw share MinAtar's one RandomState, as in the package), and its own action
  sampler numpy.random.default_rng([seed, e]). So results do not depend on how episodes are split over workers.
- Report: mean, se (= population std / sqrt(n)), episodes, max, mean_length, and per-episode returns/lengths.
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games import CHANNELS, NPZ_KEYS  # noqa: E402


DTYPE = np.float32  # the forward pass precision (--float64 only for checks)


def load_net(path):
    d = np.load(path)
    return [np.asarray(d[NPZ_KEYS[k]], dtype=np.float32).astype(DTYPE) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]]


def run_chunk(args):
    game, net_path, episodes, seed, max_steps, dtype = args
    global DTYPE
    DTYPE = np.dtype(dtype).type
    from minatar import Environment
    W1, b1, W2, b2, W3, b3 = load_net(net_path)
    probe = Environment(game, sticky_action_prob=0.1)
    names = probe.env.channels  # name -> index in state()
    n_src = probe.state_shape()[2]
    M = np.zeros((n_src, 6), dtype=np.float32)
    for k, srcs in enumerate(CHANNELS[game]):
        for s in srcs:
            M[names[s], k] = 1.0
    envs, rngs = {}, {}
    for e in episodes:
        env = Environment(game, sticky_action_prob=0.1)
        env.seed(seed * 1_000_003 + e)
        env.reset()
        envs[e], rngs[e] = env, np.random.default_rng([seed, e])
    ret = {e: 0.0 for e in episodes}
    length = {e: 0 for e in episodes}
    active = list(episodes)
    while active:
        X = np.empty((len(active), 600), dtype=DTYPE)
        for i, e in enumerate(active):
            s = envs[e].state().reshape(100, n_src).astype(np.float32)
            X[i] = ((s @ M) > 0).reshape(600)
        h1 = np.tanh(X @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        logits = (h2 @ W3 + b3).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        cdf = np.cumsum(p, axis=1)
        nxt = []
        for i, e in enumerate(active):
            u = rngs[e].random()
            a = int(min(np.searchsorted(cdf[i], u * cdf[i, -1], side="right"), 5))
            r, done = envs[e].act(a)
            ret[e] += float(r)
            length[e] += 1
            if not done and length[e] < max_steps:
                nxt.append(e)
        active = nxt
    return [(e, ret[e], length[e]) for e in episodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("net")
    ap.add_argument("out")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--float64", action="store_true", help="forward pass in float64 (check only)")
    a = ap.parse_args()
    t0 = time.time()
    eps = list(range(a.episodes))
    chunks = [eps[i::a.workers] for i in range(a.workers)]
    jobs = [(a.game, a.net, c, a.seed, a.max_steps, "float64" if a.float64 else "float32") for c in chunks if c]
    if len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            res = pool.map(run_chunk, jobs)
    else:
        res = [run_chunk(jobs[0])]
    rows = sorted(x for r in res for x in r)
    R = np.array([x[1] for x in rows])
    L = np.array([x[2] for x in rows])
    out = {"game": a.game, "net": os.path.abspath(a.net), "mean": float(R.mean()),
           "se": float(R.std() / np.sqrt(len(R))), "episodes": len(R), "max": float(R.max()),
           "mean_length": float(L.mean()), "cut_at_max_steps": int((L >= a.max_steps).sum()),
           "seed": a.seed, "float64": a.float64, "max_steps": a.max_steps, "sticky_action_prob": 0.1, "wall_s": round(time.time() - t0, 1),
           "returns": R.tolist(), "lengths": L.tolist()}
    json.dump(out, open(a.out, "w"))
    print(f"{a.game}: mean {out['mean']:.3f} +- {out['se']:.3f} over {len(R)} episodes "
          f"(max {out['max']:.0f}, mean length {out['mean_length']:.1f}, {out['wall_s']} s)")


if __name__ == "__main__":
    main()
