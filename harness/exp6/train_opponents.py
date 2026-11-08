"""Train the Go opponents (not released by the original authors): small 243 -> 32 -> 32 -> 82 policies by self-play
on pgx go_9x9, 20,000 iterations of 1,024 games each (jaxenv, GPU).

  ppo_s{0,1,2}        PPO   (s1 = practice "strong"; s0, s2 = hidden scoring opponents)
  reinforce_s{0,1,2}  REINFORCE (s1 = practice "weak"; s0, s2 = hidden, used in the main_score like the original)

What is known from the original (see README.md): 243-32-32-82, PPO, 20,000 iterations x 1,024 games, self-play with an
opponent pool ("opponent pool refilled" on resume), checkpoints iter<k>.npz every 1,000 iterations with keys
actor/dense1/kernel ... actor/logits/kernel (the authors' PPO code, also used for Exp 1/5, is CleanRL/PureJaxRL style:
tanh, orthogonal init, separate actor/critic, Adam eps 1e-5, grad-norm 0.5, 4 epochs x 8 minibatches, clip 0.2,
entropy 0.01, value 0.5, GAE 0.99/0.95, linear lr anneal); the untuned learning rate of their Go PPO script was 3e-4.
The unpublished parts (opponent pool, which moves are trained on, REINFORCE details, ...) are chosen by a calibration
search against released data (calibration.py, README.md "Recipe search"); VARIANTS below lists every recipe tried.

Each run lives in train/<variant>/<name>/ and is resumable exactly: state.pkl (params, Adam state, RNG key, pool,
iteration) every 250 iterations; iter<k>.npz every 1,000 with a quick calibration (eval_log.jsonl).
--stop-at N trains the first N iterations of the full 20,000-iteration schedule (same lr schedule), so a screened
variant can later be continued to 20,000 without any difference. A full calibration (calib_<iter>.json, 1,000 games
against each reference) is written at the stop / end. The GPU lock (~/wsp/gpu.lock) is taken per chunk of --chunk
iterations (or not at all with --no-lock, when the caller holds it); lock-holding time is logged to
logs/gpu_time.jsonl.

  python train_opponents.py --variant v2 --name ppo_s1 [--stop-at 5000]
  python train_opponents.py --variant v2 --all --export   # the six networks; export to ~/wsp/exp6_data/opponents/
  python train_opponents.py --variant v2 --name ppo_s9 --bench 30
"""
import argparse
import fcntl
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibration as C  # noqa: E402
import gocore as G  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

DATA = os.path.expanduser("~/wsp/exp6_data")
TRAIN = os.path.join(DATA, "train")
OPP = os.path.join(DATA, "opponents")
LOCK = os.path.expanduser("~/wsp/gpu.lock")
GPU_LOG = os.path.join(DATA, "logs", "gpu_time.jsonl")
ALL = ["ppo_s1", "reinforce_s1", "ppo_s0", "ppo_s2", "reinforce_s0", "reinforce_s2"]   # practice ones first

CFG = dict(
    ITERS=20_000, GAMES=1_024, HIDDEN=32, DEPTH=2, ACTIVATION="tanh",
    INIT="orthogonal: hidden gain sqrt(2), actor output 0.01, critic output 1.0; biases 0",
    LR=3e-4, ANNEAL_LR=True, ADAM_EPS=1e-5, MAX_GRAD_NORM=0.5,
    # PPO
    UPDATE_EPOCHS=4, NUM_MINIBATCHES=8, GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5,
    # REINFORCE: one gradient step per iteration on the trained moves, return = final outcome (+1/-1) of the mover,
    # minus the batch-mean outcome as baseline; same entropy bonus
    # opponents: each game's opponent is the current policy with probability P_CURRENT (None: uniform over the
    # current policy and the filled pool slots), else a uniformly drawn snapshot of the pool (ring buffer of POOL_SIZE,
    # one snapshot added every POOL_EVERY iterations)
    POOL_SIZE=16, POOL_EVERY=100, P_CURRENT=None,
    # BOTH_SIDES: in games against the current policy, both players' moves are training data (true self-play);
    # otherwise only the learner's moves
    BOTH_SIDES=False,
    REWARD="+1 win / -1 loss at the end of the game (area score, komi 7.5; superko repeat loses)",
)

# Every recipe tried in the calibration search (README.md "Recipe search"). v1 = the first guess (Exp 1/5 defaults).
VARIANTS = {
    "v1": {},
    "v2": {"BOTH_SIDES": True, "P_CURRENT": 0.5},                       # true self-play data, recent opponents
    "v3": {"BOTH_SIDES": True, "P_CURRENT": 0.5, "ENT_COEF": 0.001},    # v2, 10x less entropy bonus
    "v4": {"BOTH_SIDES": True, "P_CURRENT": 0.5, "GAMMA": 1.0},         # v2, no discount (terminal reward only)
    "v5": {"BOTH_SIDES": True, "P_CURRENT": 0.5, "ACTIVATION": "relu"},  # v2, ReLU
    "v6": {"BOTH_SIDES": True, "P_CURRENT": 0.5, "LR": 1e-3},           # v2, larger learning rate
    "v7": {"BOTH_SIDES": True, "P_CURRENT": 1.0},                       # pure self-play (no pool)
    # round 2 (full 20,000-iteration runs; v4 overshot by iteration 10,000)
    "v8": {"BOTH_SIDES": True, "P_CURRENT": 0.5, "GAMMA": 1.0, "LR": 1e-4},   # v4 with a 3x smaller learning rate
}


# --------------------------------------------------------------------------------------------- model
def init_mlp(key, dims, out_gain):
    p = {}
    names = [f"dense{k}" for k in range(1, len(dims) - 1)]
    keys = jax.random.split(key, len(dims) - 1)
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        gain = out_gain if i == len(dims) - 2 else np.sqrt(2)
        name = names[i] if i < len(names) else None
        p[name or "last"] = {"kernel": jax.nn.initializers.orthogonal(gain)(keys[i], (a, b), jnp.float32),
                             "bias": jnp.zeros((b,), jnp.float32)}
    return p


def init_params(key, hidden, depth):
    ka, kc = jax.random.split(key)
    dims = [243] + [hidden] * depth
    actor = init_mlp(ka, dims + [82], 0.01)
    actor["logits"] = actor.pop("last")
    critic = init_mlp(kc, dims + [1], 1.0)
    critic["value"] = critic.pop("last")
    return {"actor": actor, "critic": critic}


def mlp(p, x, depth, head, act="tanh"):
    f = {"tanh": jnp.tanh, "relu": jax.nn.relu}[act]
    h = x
    for k in range(1, depth + 1):
        h = f(h @ p[f"dense{k}"]["kernel"] + p[f"dense{k}"]["bias"])
    return h @ p[head]["kernel"] + p[head]["bias"]


def actor_layers(actor, depth):
    return [(np.asarray(actor[f"dense{k}"]["kernel"]), np.asarray(actor[f"dense{k}"]["bias"]))
            for k in range(1, depth + 1)] + [(np.asarray(actor["logits"]["kernel"]), np.asarray(actor["logits"]["bias"]))]


def flat(params):
    out = {}
    for part, d in params.items():
        for layer, t in d.items():
            for k, v in t.items():
                out[f"{part}/{layer}/{k}"] = np.asarray(v)
    return out


def net_of(params, cfg):
    return {"layers": actor_layers(jax.device_get(params["actor"]), cfg["DEPTH"]), "act": cfg["ACTIVATION"]}


# --------------------------------------------------------------------------------------------- one iteration
def make_iteration(algo, cfg, opt):
    n, depth, act = cfg["GAMES"], cfg["DEPTH"], cfg["ACTIVATION"]
    gamma, lam = cfg["GAMMA"], cfg["GAE_LAMBDA"]

    def masked_logits(actor, obs, legal):
        return jnp.where(legal, mlp(actor, obs, depth, "logits", act), -1e9)

    def rollout(params, pool, key):
        """n games: the learner against the current policy or pool snapshots; the learner is Black at random."""
        k_col, k_opp, k_cur, k_play = jax.random.split(key, 4)
        learner_black = jax.random.bernoulli(k_col, 0.5, (n,))
        filled = jnp.minimum(pool["count"], cfg["POOL_SIZE"])
        if cfg["P_CURRENT"] is None:     # uniform over current + filled slots (v1)
            opp_idx = jax.random.randint(k_opp, (n,), 0, filled + 1)
        else:
            snap = jax.random.randint(k_opp, (n,), 1, filled + 1)
            opp_idx = jnp.where(jax.random.bernoulli(k_cur, cfg["P_CURRENT"], (n,)), 0, snap)
        selfplay = opp_idx == 0
        x0 = jax.vmap(lambda _: G.GAME.init())(jnp.arange(n))

        def ply(carry, k):
            x, done = carry
            obs = jax.vmap(G.encode)(x)
            legal = jax.vmap(G.GAME.legal_action_mask)(x)
            l_cur = masked_logits(params["actor"], obs, legal)
            l_pool = jax.vmap(lambda a: masked_logits(a, obs, legal))(pool["actor"])       # (P, n, 82)
            l_opp = jnp.where(selfplay[:, None], l_cur, l_pool[jnp.maximum(opp_idx - 1, 0), jnp.arange(n)])
            learner_moves = (x.color == 0) == learner_black
            lg = jnp.where(learner_moves[:, None], l_cur, l_opp)
            mv = jax.random.categorical(k, lg).astype(jnp.int32)
            logp = jax.nn.log_softmax(l_cur)[jnp.arange(n), mv]
            value = mlp(params["critic"], obs, depth, "value", act)[:, 0]
            nx = jax.vmap(G.GAME.step)(x, mv)
            nx = jax.tree_util.tree_map(
                lambda new, old: jnp.where(done.reshape((-1,) + (1,) * (new.ndim - 1)), old, new), nx, x)
            ndone = done | jax.vmap(G.GAME.is_terminal)(nx)
            trained = learner_moves | (selfplay & cfg["BOTH_SIDES"])
            rec = {"obs": obs.astype(jnp.uint8), "legal": legal, "action": mv, "logp": logp, "value": value,
                   "mask": trained & ~done, "learner": learner_moves}
            return (nx, ndone), rec

        (x, _), traj = jax.lax.scan(ply, (x0, jnp.zeros(n, bool)), jax.random.split(k_play, G.N_PLY))
        rewards = jax.vmap(G.GAME.rewards)(x)
        z = jnp.where(learner_black, rewards[:, 0], rewards[:, 1])
        traj["z_mover"] = jnp.where(traj["learner"], z[None, :], -z[None, :])      # outcome for the mover of each ply
        return traj, z, opp_idx

    def gae(traj):
        """GAE over each player's own moves (two interleaved sequences: plies of the same parity); a player's
        outcome is the reward of its last trained move."""
        def back(carry, t):
            next_v, next_adv, seen = carry                       # (n, 2): per colour = ply parity
            c = t["ply"] % 2
            nv, na, sn = next_v[:, c], next_adv[:, c], seen[:, c]
            m, v, zm = t["mask"], t["value"], t["z_mover"]
            delta = jnp.where(sn, gamma * nv - v, zm - v)
            adv = jnp.where(sn, delta + gamma * lam * na, delta)
            next_v = next_v.at[:, c].set(jnp.where(m, v, nv))
            next_adv = next_adv.at[:, c].set(jnp.where(m, adv, na))
            seen = seen.at[:, c].set(sn | m)
            return (next_v, next_adv, seen), jnp.where(m, adv, 0.0)
        init = (jnp.zeros((n, 2)), jnp.zeros((n, 2)), jnp.zeros((n, 2), bool))
        xs = {"mask": traj["mask"], "value": traj["value"], "z_mover": traj["z_mover"], "ply": jnp.arange(G.N_PLY)}
        _, adv = jax.lax.scan(back, init, xs, reverse=True)
        return adv, adv + traj["value"]

    def policy_terms(actor, b):
        lg = masked_logits(actor, b["obs"].astype(jnp.float32), b["legal"])
        logp_all = jax.nn.log_softmax(lg)
        logp = jnp.take_along_axis(logp_all, b["action"][..., None], -1)[..., 0]
        p = jnp.exp(logp_all)
        ent = -jnp.sum(jnp.where(b["legal"], p * logp_all, 0.0), -1)
        return logp, ent

    def ppo_loss(params, b):
        m = b["mask"].astype(jnp.float32)
        cnt = jnp.maximum(m.sum(), 1.0)
        logp, ent = policy_terms(params["actor"], b)
        adv = b["adv"]
        mu = (adv * m).sum() / cnt
        sd = jnp.sqrt((((adv - mu) ** 2) * m).sum() / cnt)
        adv = (adv - mu) / (sd + 1e-8)
        ratio = jnp.exp(logp - b["logp"])
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg["CLIP_EPS"], 1 + cfg["CLIP_EPS"]) * adv)
        v = mlp(params["critic"], b["obs"].astype(jnp.float32), depth, "value", act)[..., 0]
        v_clip = b["value"] + jnp.clip(v - b["value"], -cfg["CLIP_EPS"], cfg["CLIP_EPS"])
        vl = 0.5 * jnp.maximum((v - b["ret"]) ** 2, (v_clip - b["ret"]) ** 2)
        pg, vl, ent = (pg * m).sum() / cnt, (vl * m).sum() / cnt, (ent * m).sum() / cnt
        return pg + cfg["VF_COEF"] * vl - cfg["ENT_COEF"] * ent, (pg, vl, ent)

    def reinforce_loss(params, b):
        m = b["mask"].astype(jnp.float32)
        cnt = jnp.maximum(m.sum(), 1.0)
        logp, ent = policy_terms(params["actor"], b)
        pg = -((logp * b["adv"]) * m).sum() / cnt
        ent = (ent * m).sum() / cnt
        return pg - cfg["ENT_COEF"] * ent, (pg, jnp.float32(0), ent)

    def iteration(params, opt_state, pool, key):
        k_roll, k_perm = jax.random.split(key)
        traj, z, opp_idx = rollout(params, pool, k_roll)
        keep = ("obs", "legal", "action", "logp", "value", "mask")
        if algo == "ppo":
            adv, ret = gae(traj)
            traj["adv"], traj["ret"] = adv, ret
            keep += ("adv", "ret")
        else:
            m = traj["mask"]
            base = (traj["z_mover"] * m).sum() / jnp.maximum(m.sum(), 1)
            traj["adv"] = traj["z_mover"] - base
            keep += ("adv",)
        # (162, n, ...) -> game-major for minibatching over games
        data = {k: jnp.swapaxes(traj[k], 0, 1) for k in keep}
        if algo == "ppo":
            nmb = cfg["NUM_MINIBATCHES"]

            def epoch(carry, k):
                params, opt_state = carry
                perm = jax.random.permutation(k, n).reshape(nmb, n // nmb)

                def mb(carry, idx):
                    params, opt_state = carry
                    b = jax.tree_util.tree_map(lambda a: a[idx], data)
                    (loss, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(params, b)
                    upd, opt_state = opt.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), (loss,) + aux
                return jax.lax.scan(mb, (params, opt_state), perm)

            (params, opt_state), stats = jax.lax.scan(epoch, (params, opt_state),
                                                      jax.random.split(k_perm, cfg["UPDATE_EPOCHS"]))
            stats = jax.tree_util.tree_map(jnp.mean, stats)
        else:
            (loss, aux), g = jax.value_and_grad(reinforce_loss, has_aux=True)(params, data)
            upd, opt_state = opt.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
            stats = (loss,) + aux
        info = {"loss": stats[0], "pg": stats[1], "vf": stats[2], "ent": stats[3],
                "win_vs_current": jnp.sum((z > 0) & (opp_idx == 0)) / jnp.maximum(jnp.sum(opp_idx == 0), 1),
                "win_vs_pool": jnp.sum((z > 0) & (opp_idx > 0)) / jnp.maximum(jnp.sum(opp_idx > 0), 1),
                "trained_moves": traj["mask"].sum() / n}
        return params, opt_state, info

    return jax.jit(iteration)


def make_opt(cfg, algo):
    steps_per_iter = cfg["UPDATE_EPOCHS"] * cfg["NUM_MINIBATCHES"] if algo == "ppo" else 1
    lr = cfg["LR"]
    if cfg["ANNEAL_LR"]:
        sched = lambda c: lr * (1.0 - (c // steps_per_iter) / cfg["ITERS"])  # noqa: E731  (PureJaxRL)
    else:
        sched = lr
    return optax.chain(optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]), optax.adam(sched, eps=cfg["ADAM_EPS"]))


# --------------------------------------------------------------------------------------------- driver
class GpuLock:
    def __init__(self, tag, use=True):
        self.tag, self.use = tag, use

    def __enter__(self):
        if self.use:
            self.f = open(LOCK, "a")
            fcntl.flock(self.f, fcntl.LOCK_EX)
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        t1 = time.time()
        if self.use:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
        os.makedirs(os.path.dirname(GPU_LOG), exist_ok=True)
        open(GPU_LOG, "a").write(json.dumps({"tag": self.tag, "start": self.t0, "end": t1, "s": round(t1 - self.t0, 1),
                                             "own_lock": self.use}) + "\n")


def full_calibration(params, cfg, d, it, log, name):
    net = net_of(params, cfg)
    ours = C.calibrate(net, games=1000, seed=31_000 + it)
    role = name if name in C.ROLES else name.rsplit("_s", 1)[0] + "_s1"
    rows, summ = C.compare(ours, role, 1000)
    fam = C.family_targets(name.split("_")[0])
    fam_d = {r: ours[r] - fam[r] for r in ours if r in fam}
    rec = {"iter": it, "ours": ours, "role_targets": rows, "role_summary": summ, "family_targets": fam,
           "family_diff": fam_d,
           "family_mean_diff_refs": float(np.mean([v for r, v in fam_d.items() if r != "random"]))}
    json.dump(rec, open(os.path.join(d, f"calib_{it}.json"), "w"), indent=1)
    log(f"{name} calibration @{it}: " + json.dumps({k: round(v, 3) for k, v in ours.items()})
        + f" family mean diff {rec['family_mean_diff_refs']:+.3f}")
    return rec


def train(name, variant="v1", iters=None, bench=0, chunk=250, ckpt=250, stop_at=None, export=False, use_lock=True,
          log=print):
    algo, seed = name.rsplit("_s", 1)
    seed = int(seed)
    cfg = dict(CFG)
    cfg.update(VARIANTS[variant])
    if iters:
        cfg["ITERS"] = iters
    d = os.path.join(TRAIN, variant, name if not bench else f"_bench_{name}")
    os.makedirs(d, exist_ok=True)
    json.dump({"name": name, "algo": algo, "seed": seed, "variant": variant, **cfg},
              open(os.path.join(d, "config.json"), "w"), indent=1)
    opt = make_opt(cfg, algo)
    step = make_iteration(algo, cfg, opt)
    sp = os.path.join(d, "state.pkl")
    if os.path.exists(sp) and not bench:
        st = pickle.load(open(sp, "rb"))
        params, opt_state = jax.tree_util.tree_map(jnp.asarray, (st["params"], st["opt_state"]))
        pool, key, it = jax.tree_util.tree_map(jnp.asarray, st["pool"]), jnp.asarray(st["key"]), st["iter"]
        log(f"{variant}/{name}: resumed at iteration {it}")
    else:
        key = jax.random.PRNGKey(seed)
        key, k0 = jax.random.split(key)
        params = init_params(k0, cfg["HIDDEN"], cfg["DEPTH"])
        opt_state = opt.init(params)
        pool = {"actor": jax.tree_util.tree_map(lambda a: jnp.zeros((cfg["POOL_SIZE"],) + a.shape, a.dtype),
                                                params["actor"]), "count": jnp.int32(0)}
        it = 0
    total = bench or min(stop_at or cfg["ITERS"], cfg["ITERS"])
    warm = False
    logf = open(os.path.join(d, "train_log.jsonl"), "a")
    t0, n0 = time.time(), it
    while it < total:
        with GpuLock(f"{variant}/{name}", use_lock):
            t0, n0 = time.time(), it
            end = min(total, it + chunk)
            while it < end:
                if it % cfg["POOL_EVERY"] == 0:
                    slot = pool["count"] % cfg["POOL_SIZE"]
                    pool = {"actor": jax.tree_util.tree_map(lambda P, a: P.at[slot].set(a), pool["actor"],
                                                            params["actor"]), "count": pool["count"] + 1}
                key, k = jax.random.split(key)
                params, opt_state, info = step(params, opt_state, pool, k)
                it += 1
                if bench and not warm:              # exclude the compilation from the benchmark
                    jax.block_until_ready(params)
                    t0, n0, warm = time.time(), it, True
                    continue
                if it % 50 == 0 or it == total or bench:
                    info = {k: float(v) for k, v in jax.device_get(info).items()}
                    info.update(iter=it, s_per_iter=round((time.time() - t0) / max(1, it - n0), 3), t=time.time())
                    logf.write(json.dumps(info) + "\n")
                    logf.flush()
                if not bench and (it % ckpt == 0 or it == total):
                    tmp = sp + ".tmp"
                    pickle.dump(jax.device_get({"params": params, "opt_state": opt_state, "pool": pool,
                                                "key": key, "iter": it}), open(tmp, "wb"))
                    os.replace(tmp, sp)
                if not bench and (it % 1000 == 0 or it == total):
                    np.savez(os.path.join(d, f"iter{it}.npz"), **flat(jax.device_get(params)))
                    q = C.calibrate(net_of(params, cfg), games=256, seed=20_000 + it,
                                    refs=["astra_b0", "same_size_s0"])
                    rec = {"iter": it, **{f"vs_{k}_256": v for k, v in q.items()}, "t": time.time()}
                    open(os.path.join(d, "eval_log.jsonl"), "a").write(json.dumps(rec) + "\n")
                    log(f"{variant}/{name} eval {json.dumps({k: (round(v, 3) if k != 't' else None) for k, v in rec.items() if k != 't'})}")
            if not bench and it == total:
                full_calibration(params, cfg, d, it, log, name)
    if bench:
        jax.block_until_ready(params)
        return (time.time() - t0) / max(1, it - n0)
    if it < cfg["ITERS"]:
        log(f"{variant}/{name}: stopped at iteration {it} of {cfg['ITERS']} (--stop-at)")
        return None
    out = os.path.join(OPP if export else os.path.join(d, "export"), name if export else "")
    G.save_net(net_of(params, cfg), out, {"source": f"{d}/iter{it}.npz", "iters": it, "algo": algo, "seed": seed,
                                          "variant": variant})
    if export:      # manifest of every exported opponent, hashed from disk (integrity.py; check_inputs.py exp6)
        import integrity as I
        I.write_opponents_manifest(OPP)
        log(f"{variant}/{name}: {OPP}/manifest.sha256 updated")
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    log(f"{variant}/{name}: done, exported to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--stop-at", type=int)
    ap.add_argument("--export", action="store_true", help="final recipe: export to ~/wsp/exp6_data/opponents/")
    ap.add_argument("--no-lock", action="store_true", help="the caller holds ~/wsp/gpu.lock")
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250)
    a = ap.parse_args()
    log = lambda m: print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    names = ALL if a.all else a.name.split(",")
    for nm in names:
        if not a.bench and os.path.exists(os.path.join(TRAIN, a.variant, nm, "DONE")):
            log(f"{a.variant}/{nm}: already done")
            if a.export and not os.path.exists(os.path.join(OPP, nm, "arch.json")):
                train(nm, a.variant, export=True, use_lock=not a.no_lock, log=log)
            continue
        r = train(nm, a.variant, iters=a.iters, bench=a.bench, chunk=a.chunk, stop_at=a.stop_at, export=a.export,
                  use_lock=not a.no_lock, log=log)
        if a.bench:
            log(f"BENCH {a.variant}/{nm}: {r:.3f} s/iteration -> {r * CFG['ITERS'] / 3600:.1f} h for {CFG['ITERS']} iterations")


if __name__ == "__main__":
    main()
