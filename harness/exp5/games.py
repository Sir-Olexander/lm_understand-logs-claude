"""Per-game settings of Exp 5, copied from the prompts (runs/<game>/message.txt, "How the 6 channels are built ...").
Stdlib only (imported by run.py under the system python3 and by the scorer under jaxenv)."""

GAMES = ["breakout", "seaquest", "asterix"]

# channel k of the network input <- OR of these MinAtar channels (names of the game's state() channels)
CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
NAMES = ["player", "danger", "reward", "projectile", "trail", "gauge"]

# Original numbers (eval/minatar_eval.json -> <game>.hand) for comparison in summaries.
ORIGINAL = {
    "breakout": {"mean": 517.411, "se": 7.416952829044303, "mean_length": 7417.202, "max": 800.0},
    "seaquest": {"mean": 53.11, "se": 1.3573821814015339, "mean_length": 1411.613, "max": 158.0},
    "asterix": {"mean": 53.178, "se": 1.023305169277215, "mean_length": 858.045, "max": 177.0},
}

# Network file keys, as in the original networks/<game>/hand.npz
NPZ_KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
            "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}


def mapping_line(game):
    """The prompt's sentence for this mapping (checked against the prompt text by check_prompt)."""
    parts = []
    for k, src in enumerate(CHANNELS[game]):
        parts.append(f"channel {k} ({NAMES[k]}) <- {', '.join(src) if src else 'nothing'}")
        # e.g. "channel 0 (player) <- sub_front, sub_back"
    return "How the 6 channels are built from the game's own channels (state() in the source): " + "; ".join(parts) + "."


def check_prompt(game, prompt):
    if mapping_line(game) not in prompt:
        raise SystemExit(f"{game}: channel mapping in games.py does not match the prompt")
