"""Score a Kinetix actor on level l/h2_car_ramp (run with ~/wsp/venvs/jaxenv/bin/python).

Environment: kinetix-env 3.0.2 (jax2d 1.0.1), the level file l/h2_car_ramp shipped with it, symbolic-flat
observation (679 numbers), multi-discrete actions (6 entries: 4 motor bindings with values 0/1/2 -> off/+1/-1,
2 thruster bindings 0/1), frame_skip 2, at most 256 steps, dense reward 0.2 x (d_prev - d) as stated in the prompt.

Actor (official Kinetix ActorCriticSymbolicRNN, non-recurrent, tanh, depth 5, width 128):
  embed = relu(x @ embed.W + embed.b); a_k = tanh(a_(k-1) @ actor<k>.W + actor<k>.b); logits = a_4 @ actor_head.W + b.
Each action entry is sampled independently from the softmax of its logits (Kinetix MultiDiscreteActionDistribution).

Outcome of an episode: success (green touches blue, +1), red (green touches red, -1; red wins a tie), invalid
(non-finite state), timeout (256 steps). Progress toward the goal: see PROGRESS below.

Usage: kinetix_eval.py --weights W.npz --stream search|test|NAME --episodes N --out result.json [--trace]
W.npz holds either the Kinetix parameter names (params/Dense_0/kernel ...) or embed.W, embed.b, actor0.W, ... .
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import logging

logging.getLogger("absl").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import numpy as np

from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_from_json_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from actor_io import load_actor  # noqa: E402

LEVEL = "l/h2_car_ramp"
DENSE_REWARD_SCALE = 0.2
MAX_STEPS = 256
DIMS = [3, 3, 3, 3, 2, 2]
# fixed seeds of the two episode streams (the original's seeds are not published; see README)
STREAM_SEEDS = {"search": 0, "test": 1}
BODY, GREEN, WHEEL = 5, 12, 13  # shape ids shown in the trace (polygon 5; circles 12, 13 = circle slots 0, 1)


def logits_fn(w, x):
    h = jax.nn.relu(x @ w["embed.W"] + w["embed.b"])
    for k in range(5):
        h = jnp.tanh(h @ w[f"actor{k}.W"] + w[f"actor{k}.b"])
    return h @ w["actor_head.W"] + w["actor_head.b"]


def split_logits(lg):
    out, i = [], 0
    for n in DIMS:
        out.append(lg[..., i:i + n])
        i += n
    return out


def make_env():
    level, static_params, env_params = load_from_json_file(LEVEL)
    env_params = env_params.replace(dense_reward_scale=DENSE_REWARD_SCALE, max_timesteps=MAX_STEPS)
    env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, reset_fn=lambda rng: level,
                           env_params=env_params, static_env_params=static_params, auto_reset=False)
    return env, env_params, level


def build_runner(env, env_params, level):
    def episode(w, key, greedy):
        obs0, st0 = env.reset(jax.random.PRNGKey(0), env_params, override_reset_state=level)

        def gpos(st):
            return st.circle.position[0]  # green wheel (shape 12)

        blue = None

        def step(carry, _):
            obs, st, key, done, out = carry
            key, k_act = jax.random.split(key)
            lg = logits_fn(w, obs)
            parts = split_logits(lg)
            ks = jax.random.split(k_act, len(DIMS))
            if greedy:
                act = jnp.stack([jnp.argmax(p) for p in parts])
            else:
                act = jnp.stack([jax.random.categorical(k, p) for k, p in zip(ks, parts)])
            probs0 = jax.nn.softmax(parts[0])
            rec = jnp.concatenate([st.polygon.position[BODY], st.polygon.rotation[BODY][None],
                                   st.circle.position[0], st.circle.position[1], act[:1].astype(jnp.float32),
                                   probs0])
            obs2, st2, r, d, info = env.step(key, st, act, env_params)
            invalid = jax.tree.reduce(jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), st2), False)
            live = ~done
            succ = d & info["GoalR"] & ~invalid
            red = d & (r < -0.5) & ~info["GoalR"] & ~invalid
            dist = info["distance"]
            out = {
                "ret": out["ret"] + jnp.where(live, r, 0.0),
                "len": out["len"] + live.astype(jnp.int32),
                "succ": out["succ"] | (live & succ),
                "red": out["red"] | (live & red),
                "invalid": out["invalid"] | (live & invalid),
                "dmin": jnp.where(live, jnp.minimum(out["dmin"], dist), out["dmin"]),
                "dlast": jnp.where(live, dist, out["dlast"]),
            }
            return (obs2, st2, key, done | d, out), rec

        init = {"ret": jnp.float32(0), "len": jnp.int32(0), "succ": False, "red": False, "invalid": False,
                "dmin": jnp.float32(1e9), "dlast": jnp.float32(0)}
        (_, _, _, _, out), recs = jax.lax.scan(step, (obs0, st0, key, jnp.bool_(False), init), None, MAX_STEPS)
        return out, recs

    run_sample = jax.jit(jax.vmap(lambda w, k: episode(w, k, False), in_axes=(None, 0)))
    run_greedy = jax.jit(lambda w: episode(w, jax.random.PRNGKey(0), True))
    return run_sample, run_greedy


def initial_distance(env, env_params, level):
    # distance green-blue in the start state (as computed by the environment's reward info)
    st = level
    green = st.circle.position[0]
    blue = st.polygon.position[4]
    return float(jnp.linalg.norm(green - blue))


def outcome(o):
    return "success" if o["succ"] else "red" if o["red"] else "invalid" if o["invalid"] else "timeout"


def trace_text(recs, length, out_name):
    rows = list(range(0, length, 16))
    if rows[-1] != length - 1:
        rows.append(length - 1)
    lines = [f"A sampled episode ({out_name} after {length} steps), every 16 steps: step | car body (shape 5) x, y, "
             f"angle | green wheel (shape 12) x, y | wheel (shape 13) x, y | action entry 0 | probabilities of entry "
             f"0's values 0 / 1 / 2"]
    for t in rows:
        r = recs[t]
        lines.append(f"{t:5d} | {r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f} | {r[3]:.2f}, {r[4]:.2f} | {r[5]:.2f}, {r[6]:.2f} | "
                     f"{int(r[7])} | {r[8]:.2f} / {r[9]:.2f} / {r[10]:.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stream", default="search")
    ap.add_argument("--seed", type=int, default=None, help="overrides the stream's seed")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default=None, help="save per-episode arrays (npz)")
    a = ap.parse_args()

    t0 = time.time()
    w = {k: jnp.asarray(v, jnp.float32) for k, v in load_actor(a.weights).items()}
    env, env_params, level = make_env()
    run_sample, run_greedy = build_runner(env, env_params, level)
    seed = STREAM_SEEDS.get(a.stream, 0) if a.seed is None else a.seed
    keys = jax.random.split(jax.random.PRNGKey(seed), a.episodes)
    outs, rec0 = [], None
    for i in range(0, a.episodes, a.batch):
        o, recs = run_sample(w, keys[i:i + a.batch])
        o = jax.tree.map(np.asarray, o)
        outs.append(o)
        if rec0 is None:
            rec0 = np.asarray(recs[0])
    o = {k: np.concatenate([x[k] for x in outs]) for k in outs[0]}
    go, grec = run_greedy(w)
    go = jax.tree.map(np.asarray, go)
    d0 = initial_distance(env, env_params, level)
    n = a.episodes
    succ, red, inval = o["succ"], o["red"] & ~o["succ"], o["invalid"] & ~o["succ"] & ~o["red"]
    tout = ~(succ | red | inval)
    prog_final = np.where(succ, 1.0, np.clip(1 - o["dlast"] / d0, 0, 1))
    prog_best = np.where(succ, 1.0, np.clip(1 - o["dmin"] / d0, 0, 1))
    res = {
        "success": float(succ.mean()), "red": float(red.mean()), "timeout": float(tout.mean()),
        "invalid": float(inval.mean()), "return": float(o["ret"].mean()), "length": float(o["len"].mean()),
        "progress": float(prog_best.mean()), "progress_final": float(prog_final.mean()),
        "argmax_success": float(bool(go["succ"])), "argmax_outcome": outcome(go),
        "episodes": n, "stream": a.stream, "seed": seed, "mode": "sample",
        "env_steps": int(o["len"].sum()), "success_count": int(succ.sum()),
        "success_se": float(np.sqrt(succ.mean() * (1 - succ.mean()) / n)),
        "d0": d0, "wall_s": round(time.time() - t0, 1),
        "jax": jax.__version__, "device": str(jax.devices()[0]),
    }
    L0 = int(o["len"][0])
    e0 = outcome({k: v[0] for k, v in o.items()})
    res["text"] = (
        f"Result over {n} episodes (actions sampled from the policy): success {int(succ.sum())} "
        f"({100 * succ.mean():.1f} %), red {int(red.sum())}, timeout {int(tout.sum())}, invalid {int(inval.sum())}; "
        f"mean return {res['return']:.3f}; mean episode length {res['length']:.0f} steps; mean progress toward the "
        f"goal {res['progress']:.3f} (1 = reached). Most-likely-action episode: {res['argmax_outcome']}.\n"
        + trace_text(rec0, L0, e0))
    json.dump(res, open(a.out, "w"), indent=1)
    if a.dump:
        np.savez(a.dump, **o)
    print(json.dumps({k: v for k, v in res.items() if k != "text"}))
    print(res["text"])


if __name__ == "__main__":
    main()
