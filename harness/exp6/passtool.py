"""The two pass tools of Experiment 6 (run by the driver in ~/wsp/venvs/jaxenv; the model never sees this code).

  python passtool.py games     <request.json> <work_dir> <k> <budget>
  python passtool.py positions <request.json> <work_dir> <k> <budget>

Reads the current network from <work_dir>/net/<name>.npy (missing tensors = zeros), writes <work_dir>/pass_<k>/ and
prints one JSON line: the result dict shown to the model ({"ok": true, ...}) or {"ok": false, "error": "..."}.
Formats copied from the original logs (runs/budget_*/turns/*/prompt.txt, work/pass_<k>/README.txt).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.expanduser("~/wsp/exp6_data")
OPP = os.environ.get("EXP6_OPPONENTS", os.path.join(DATA, "opponents"))   # override only for validation
PRACTICE = {"random": "random", "weak": "reinforce_s1", "strong": "ppo_s1"}
ARCH = {"depth": 6, "width": 512}
MAX_GAMES, MAX_POSITIONS = 128, 256

GAMES_README = ("games.json: one entry per game (opponent, net_color, result +1/-1, margin = net's area score minus "
                "the opponent's incl. komi, end, stones captured / lost, moves: ply, by, move (row*9+col, 81 = pass), "
                "and for the network's moves its top-5 [move, probability]). boards.npy: (games, 162, 81) int8, the "
                "board AFTER each ply (1 Black, -1 White, 0 empty; index row*9+col); valid.npy: (games, 162) which "
                "plies were played.")
POS_README = ("inputs.npy (P, 243) the network inputs; legal.npy (P, 82) the legal moves (for 'board' positions: the "
              "empty points and pass; suicide and ko are not checked); logits.npy, probs.npy (P, 82); ")
POS_README_ACT = POS_README + "act_L1.npy .. act_L6.npy (P, 512) the hidden activations after ReLU."
POS_README_NOACT = POS_README + 'no activations (ask with "activations": true).'


class BadRequest(Exception):
    pass


def check_games(req):
    if not isinstance(req, dict) or set(req) != {"games"} or not isinstance(req["games"], dict) or not req["games"]:
        raise BadRequest('expected {"games": {"strong": n, "weak": n, "random": n}} (any subset of the opponents)')
    total = 0
    for name, n in req["games"].items():
        if name not in PRACTICE:
            raise BadRequest(f"unknown opponent {name!r}; the opponents are 'random', 'weak' and 'strong'")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0 or n % 2:
            raise BadRequest(f"the number of games against {name!r} must be an even number >= 0, got {n!r}")
        total += n
    if total == 0 or total > MAX_GAMES:
        raise BadRequest(f"between 2 and {MAX_GAMES} games in total, got {total}")


def check_positions(req):
    if not isinstance(req, dict) or "positions" not in req or set(req) - {"positions", "activations"}:
        raise BadRequest('expected {"positions": [...], "activations": true or false}')
    ps = req["positions"]
    if not isinstance(ps, list) or not 1 <= len(ps) <= MAX_POSITIONS:
        raise BadRequest(f"positions must be a list of 1 to {MAX_POSITIONS} positions")
    if not isinstance(req.get("activations", False), bool):
        raise BadRequest("activations must be true or false")
    for i, p in enumerate(ps):
        if not isinstance(p, dict) or not (set(p) in ({"board", "to_move"}, {"moves"})):
            raise BadRequest(f'position {i}: expected {{"board": [...], "to_move": ...}} or {{"moves": [...]}}')
        if "moves" in p and not isinstance(p["moves"], list):
            raise BadRequest(f"position {i}: moves must be a list")


def label(m):
    return "pass" if m == 81 else f"({m // 9},{m % 9})"


def games_pass(req, work, k):
    import numpy as np
    import gocore as G
    import integrity as I
    net = G.load_net(os.path.join(work, "net"), ARCH)
    out_dir = os.path.join(work, f"pass_{k}")
    os.makedirs(out_dir, exist_ok=True)
    per, games, boards, valid = {}, [], [], []
    for j, (name, n) in enumerate(sorted(req["games"].items())):     # summaries in name order, as in the logs
        if n == 0:
            continue
        if PRACTICE[name] == "random":
            opp = G.RANDOM
        else:
            I.verify_opponent(PRACTICE[name], OPP)          # against manifest.sha256 before loading
            opp = G.load_net(os.path.join(OPP, PRACTICE[name]))
        r = G.play(net, opp, n, seed=1_000_003 * k + 7919 * j, record=True)
        per[name] = G.summary(r)
        for g in range(n):
            moves = []
            for t in range(G.N_PLY):
                if not r["valid"][g, t]:
                    break
                e = {"ply": t, "by": "net" if r["a_moves"][g, t] else "opponent", "move": int(r["moves"][g, t])}
                if r["a_moves"][g, t]:
                    e["top5"] = [[int(m), round(float(p), 4)] for m, p in zip(r["top_i"][g, t], r["top_p"][g, t])]
                moves.append(e)
            # keys and values as in the original games.json (seen in Astra's printed records, runs/budget_*/)
            games.append({"game": len(games), "opponent": name, "net_color": "black" if r["a_black"][g] else "white",
                          "result": int(r["result"][g]), "margin": float(r["margin"][g]),
                          "end": G.END_NAMES[int(r["end"][g])], "stones_captured_by_net": int(r["captured"][g]),
                          "stones_lost_by_net": int(r["lost"][g]), "moves": moves})
        boards.append(np.where(r["valid"][..., None], r["board"], 0).astype(np.int8))
        valid.append(r["valid"])
    json.dump(games, open(os.path.join(out_dir, "games.json"), "w"))
    np.save(os.path.join(out_dir, "boards.npy"), np.concatenate(boards))
    np.save(os.path.join(out_dir, "valid.npy"), np.concatenate(valid))
    json.dump(per, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    open(os.path.join(out_dir, "README.txt"), "w").write(GAMES_README)
    return {"ok": True, "pass": k, "kind": "games", "per_opponent": per, "files": f"/work/pass_{k}/",
            "readme": GAMES_README}


def positions_pass(req, work, k):
    import numpy as np
    import gocore as G
    xs, legals = [], []
    for i, p in enumerate(req["positions"]):
        try:
            x, legal = G.replay(p["moves"]) if "moves" in p else G.board_input(p["board"], p["to_move"])
        except ValueError as e:
            raise BadRequest(f"position {i}: {e}") from None
        xs.append(x)
        legals.append(legal)
    X, L = np.stack(xs).astype(np.float32), np.stack(legals).astype(bool)
    net = G.load_net(os.path.join(work, "net"), ARCH)
    logits, acts = G.np_forward(net, X)
    z = np.where(L, logits, -np.inf)
    z = z - z.max(1, keepdims=True)
    probs = np.where(L, np.exp(z), 0)
    probs = (probs / probs.sum(1, keepdims=True)).astype(np.float32)
    top3 = []
    for i in range(len(X)):
        order = np.argsort(-probs[i], kind="stable")[:3]
        top3.append([[label(int(m)), round(float(probs[i, m]), 4)] for m in order])
    out_dir = os.path.join(work, f"pass_{k}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "inputs.npy"), X)
    np.save(os.path.join(out_dir, "legal.npy"), L)
    np.save(os.path.join(out_dir, "logits.npy"), logits.astype(np.float32))
    np.save(os.path.join(out_dir, "probs.npy"), probs)
    act = bool(req.get("activations", False))
    if act:
        for j, a in enumerate(acts, 1):
            np.save(os.path.join(out_dir, f"act_L{j}.npy"), a.astype(np.float32))
    readme = POS_README_ACT if act else POS_README_NOACT
    open(os.path.join(out_dir, "README.txt"), "w").write(readme)
    return {"ok": True, "pass": k, "kind": "positions", "positions": len(X), "top3_per_position": top3,
            "files": f"/work/pass_{k}/", "readme": readme}


def main():
    kind, req_path, work, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    try:
        try:
            req = json.load(open(req_path))
        except ValueError as e:
            raise BadRequest(f"{type(e).__name__}: {e}") from None
        (check_games if kind == "games" else check_positions)(req)
        res = (games_pass if kind == "games" else positions_pass)(req, work, k)
    except BadRequest as e:
        res = {"ok": False, "error": str(e)}
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
