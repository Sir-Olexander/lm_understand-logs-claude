"""9x9 Go engine shared by the pass tools, the scorer and the opponent training (runs in ~/wsp/venvs/jaxenv).

Rules: pgx 2.6.0 `go_9x9` game logic (pgx._src.games.go.Game, size 9, komi 7.5, 162-move cap, positional superko
ends the game with a loss for the player who made the repeating move, area scoring). We call the Game class directly
(not pgx.make's Env wrapper) so we choose who is Black and build the 243-number input of the task prompt:
input (row*9+col)*3+plane, plane 0 = mover's stones, plane 1 = opponent's stones, plane 2 = 1 when White is to move.
This equals pgx's own observation with history_length=1 (planes: mover, opponent, colour).

A network is {"layers": [(W, b), ...], "act": "relu" | "tanh"}: h = act(x @ W + b) for the hidden layers, the last
layer is linear (82 logits: row*9+col, 81 = pass). "random" is the uniform random legal player.
Illegal moves are masked and moves are sampled from the softmax of the remaining logits (temperature 1).
"""
import functools
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from pgx._src.games.go import Game, _count_scores  # noqa: E402

_CACHE = os.path.expanduser("~/wsp/exp6_data/jax_cache")
try:
    os.makedirs(_CACHE, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.0)
except Exception:  # noqa: BLE001
    pass

SIZE, N_PLY, KOMI = 9, 162, 7.5
GAME = Game(size=SIZE, komi=KOMI, history_length=1)   # max_termination_steps defaults to 9*9*2 = 162
assert GAME.max_termination_steps == N_PLY
END_NAMES = ("two passes", "repetition", "move limit")   # the labels in the original games.json
RANDOM = "random"


# ----------------------------------------------------------------------------------------------- networks
def load_net(d, arch=None, prefix=""):
    """Network from a folder of L1.W.npy ... out.b.npy (+ arch.json: depth, width, optional act; default relu).
    Tensors that do not exist are zeros (as for the written network: tensors never saved stay zero)."""
    a = dict(arch or json.load(open(os.path.join(d, "arch.json"))))
    depth, width, act = a["depth"], a["width"], a.get("act", "relu")
    layers = []
    for k in range(1, depth + 2):
        name = f"L{k}" if k <= depth else "out"
        shape = (243 if k == 1 else width, width if k <= depth else 82)
        w = os.path.join(d, prefix + name + ".W.npy")
        b = os.path.join(d, prefix + name + ".b.npy")
        W = np.load(w).astype(np.float32) if os.path.exists(w) else np.zeros(shape, np.float32)
        B = np.load(b).astype(np.float32) if os.path.exists(b) else np.zeros(shape[1], np.float32)
        assert W.shape == shape and B.shape == (shape[1],), (d, name, W.shape, B.shape)
        layers.append((W, B))
    return {"layers": layers, "act": act}


def save_net(net, d, extra_arch=None):
    """Save with verified writes (integrity.py: fsync, drop cache, re-read and compare the SHA-256)."""
    import integrity as I
    os.makedirs(d, exist_ok=True)
    layers = net["layers"]
    depth = len(layers) - 1
    for k, (W, B) in enumerate(layers, 1):
        name = f"L{k}" if k <= depth else "out"
        I.write_verified(os.path.join(d, name + ".W.npy"), I.npy_bytes(np.asarray(W, np.float32)))
        I.write_verified(os.path.join(d, name + ".b.npy"), I.npy_bytes(np.asarray(B, np.float32)))
    arch = {"depth": depth, "width": int(np.asarray(layers[0][0]).shape[1]), "act": net["act"]}
    arch.update(extra_arch or {})
    I.write_verified(os.path.join(d, "arch.json"), json.dumps(arch).encode())


def _act(name):
    return {"relu": jax.nn.relu, "tanh": jnp.tanh}[name]


def logits_of(layers, act, x):
    if act == RANDOM:
        return jnp.zeros(x.shape[:-1] + (82,), jnp.float32)
    f = _act(act)
    h = x
    for W, b in layers[:-1]:
        h = f(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


def np_forward(net, x):
    """Numpy forward (positions pass): logits and the hidden activations after the nonlinearity."""
    f = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh}[net["act"]]
    h, acts = np.asarray(x, np.float32), []
    for W, b in net["layers"][:-1]:
        h = f(h @ W + b).astype(np.float32)
        acts.append(h)
    W, b = net["layers"][-1]
    return (h @ W + b).astype(np.float32), acts


def as_player(net):
    """(layers as a jax pytree, activation name) for play()."""
    if net == RANDOM or net is None:
        return (), RANDOM
    return tuple((jnp.asarray(W), jnp.asarray(b)) for W, b in net["layers"]), net["act"]


# ----------------------------------------------------------------------------------------------- game
def encode(x):
    """243 inputs from the point of view of the player to move."""
    color = x.color
    sign = jnp.where(color == 0, 1, -1)
    b = jnp.sign(x.board) * sign
    planes = jnp.stack([b > 0, b < 0, jnp.full((SIZE * SIZE,), color == 1)], -1)
    return planes.reshape(-1).astype(jnp.float32)


def end_kind(x):
    """0 two passes, 1 repetition (positional superko), 2 move limit."""
    return jnp.where(x.is_psk, 1, jnp.where(x.consecutive_pass_count >= 2, 0, 2))


@functools.partial(jax.jit, static_argnames=("act_a", "act_b", "record"))
def _play(layers_a, layers_b, key, a_black, *, act_a, act_b, record):
    n = a_black.shape[0]
    x0 = jax.vmap(lambda _: GAME.init())(jnp.arange(n))
    done0 = jnp.zeros(n, bool)

    def ply(carry, k):
        x, done = carry
        obs = jax.vmap(encode)(x)
        legal = jax.vmap(GAME.legal_action_mask)(x)
        la = logits_of(layers_a, act_a, obs)
        lb = logits_of(layers_b, act_b, obs)
        a_moves = (x.color == 0) == a_black
        lg = jnp.where(a_moves[:, None], la, lb)
        lg = jnp.where(legal, lg, -jnp.inf)
        keys = jax.random.split(k, n)
        mv = jax.vmap(jax.random.categorical)(keys, lg).astype(jnp.int32)
        nx = jax.vmap(GAME.step)(x, mv)
        nx = jax.tree_util.tree_map(
            lambda new, old: jnp.where(done.reshape((-1,) + (1,) * (new.ndim - 1)), old, new), nx, x)
        ndone = done | jax.vmap(GAME.is_terminal)(nx)
        out = {"move": mv, "a_moves": a_moves, "valid": ~done}
        if record:
            p = jax.nn.softmax(lg, axis=-1)
            tp, ti = jax.lax.top_k(p, 5)
            out.update(board=jnp.clip(nx.board, -1, 1).astype(jnp.int8), top_p=tp, top_i=ti.astype(jnp.int16))
        return (nx, ndone), out

    (x, done), rec = jax.lax.scan(ply, (x0, done0), jax.random.split(key, N_PLY))
    scores = jax.vmap(lambda s: _count_scores(s, SIZE))(x)            # (n, 2) area: black, white
    rewards = jax.vmap(GAME.rewards)(x)                                 # (n, 2) +-1: black, white
    fin = {"rewards": rewards, "scores": scores, "end": jax.vmap(end_kind)(x), "length": x.step_count,
           "captured": x.num_captured, "terminal": done}
    return fin, rec


def play(net_a, net_b, n, seed, a_black=None, record=False, chunk=None):
    """n games of A against B (A Black in the first half unless a_black is given). Returns per-game numpy results.
    Ply-major records (162, n, ...) are transposed to game-major."""
    la, aa = as_player(net_a)
    lb, ab = as_player(net_b)
    if a_black is None:
        a_black = np.arange(n) < n // 2
    a_black = np.asarray(a_black, bool)
    chunk = chunk or n
    parts = []
    key = jax.random.PRNGKey(seed)
    for i0 in range(0, n, chunk):
        key, sub = jax.random.split(key)
        fin, rec = _play(la, lb, sub, jnp.asarray(a_black[i0:i0 + chunk]), act_a=aa, act_b=ab, record=record)
        parts.append(jax.device_get((fin, rec)))
    fin = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    rec = {k: np.concatenate([np.swapaxes(p[1][k], 0, 1) for p in parts]) for k in parts[0][1]}
    assert fin["terminal"].all()
    col = np.where(a_black, 0, 1)
    res = {
        "a_black": a_black,
        "result": fin["rewards"][np.arange(n), col].astype(np.int32),               # +1 win / -1 loss for A
        "margin": np.where(a_black, 1, -1) * (fin["scores"][:, 0] - fin["scores"][:, 1] - KOMI),
        "end": fin["end"], "length": fin["length"],
        "captured": fin["captured"][np.arange(n), col], "lost": fin["captured"][np.arange(n), 1 - col],
        "moves": rec["move"], "a_moves": rec["a_moves"], "valid": rec["valid"],
    }
    a_valid = rec["a_moves"] & rec["valid"]
    a_pass = a_valid & (rec["move"] == 81)
    res["a_passes"] = a_pass.sum(1)
    res["a_pass_before_60"] = a_pass[:, :60].any(1)
    if record:
        res.update(board=rec["board"], top_p=rec["top_p"], top_i=rec["top_i"])
    return res


def summary(r, idx=None):
    """The per-opponent summary of the original (score etc. as float32 means, like the released files)."""
    if idx is not None:
        r = {k: v[idx] for k, v in r.items()}
    n = len(r["result"])
    win = r["result"] > 0
    blk = r["a_black"]
    f32 = lambda a: float(np.mean(np.asarray(a, np.float32))) if len(a) else 0.0  # noqa: E731
    return {
        "games": int(n), "score": f32(win), "wins": int(win.sum()), "losses": int((~win).sum()),
        "score_as_black": f32(win[blk]), "score_as_white": f32(win[~blk]),
        "mean_margin": f32(r["margin"]), "mean_length": float(np.mean(r["length"])),
        "stones_captured_per_game": float(np.mean(r["captured"])),
        "stones_lost_per_game": float(np.mean(r["lost"])),
        "passes_per_game": float(np.mean(r["a_passes"])),
        "games_with_a_pass_before_move_60": int(r["a_pass_before_60"].sum()),
        "end_two_passes": int((r["end"] == 0).sum()), "end_repetition": int((r["end"] == 1).sum()),
        "end_move_limit": int((r["end"] == 2).sum()),
    }


# ----------------------------------------------------------------------------------------------- positions
def replay(moves):
    """Play a list of moves (row*9+col or 'pass') from the empty board with exact legality.
    Returns (inputs (243,), legal (82,)) for the position after the last move."""
    x = GAME.init()
    for i, m in enumerate(moves):
        if isinstance(m, str):
            if m.strip().lower() != "pass":
                raise ValueError(f"move {i}: {m!r} is not a point index or 'pass'")
            a = 81
        elif isinstance(m, (int, np.integer)) and not isinstance(m, bool) and 0 <= int(m) <= 81:
            a = int(m)
        else:
            raise ValueError(f"move {i}: {m!r} is not a point index 0..80 or 'pass'")
        if bool(GAME.is_terminal(x)):
            raise ValueError(f"move {i}: the game is already over")
        if not bool(_legal_jit(x)[a]):
            raise ValueError(f"move {i}: {m!r} is illegal")
        x = _step_jit(x, jnp.int32(a))
    if bool(GAME.is_terminal(x)):
        raise ValueError("the game is over after these moves")
    return np.asarray(_encode_jit(x)), np.asarray(_legal_jit(x))


_legal_jit = jax.jit(GAME.legal_action_mask)
_step_jit = jax.jit(GAME.step)
_encode_jit = jax.jit(encode)


def board_input(rows, to_move):
    """Inputs and the legal moves (empty points and pass) for a 'board' position."""
    if not isinstance(rows, list) or len(rows) != 9 or not all(isinstance(r, str) and len(r) == 9 for r in rows):
        raise ValueError("board must be 9 strings of 9 characters")
    if to_move not in ("black", "white"):
        raise ValueError('to_move must be "black" or "white"')
    b = np.zeros(81, np.int8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch not in "XO.":
                raise ValueError(f"board character {ch!r} at ({r},{c}); use X, O or .")
            b[r * 9 + c] = {"X": 1, "O": -1, ".": 0}[ch]
    sign = 1 if to_move == "black" else -1
    x = np.stack([b * sign > 0, b * sign < 0, np.full(81, to_move == "white")], -1).reshape(-1).astype(np.float32)
    legal = np.append(b == 0, True)
    return x, legal
