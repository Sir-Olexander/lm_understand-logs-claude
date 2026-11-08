"""Load a Kinetix actor from .npz (numpy only): either the Kinetix parameter names of the official
ActorCriticSymbolicRNN (params/Dense_0 = embed, GeneralActorCriticRNN_0/DenseAndActivation_0..4 = actor0..4,
GeneralActorCriticRNN_0/Dense_0 = actor_head; the value head DenseAndActivation_5..9 / Dense_1 is ignored)
or the prompt's names (embed.W, embed.b, actor0.W, ..., actor_head.b)."""
import numpy as np

NAMES = ["embed", "actor0", "actor1", "actor2", "actor3", "actor4", "actor_head"]


def load_actor(path):
    d = np.load(path)
    if "embed.W" in d.files:
        return {k: np.asarray(d[k], np.float64) for k in d.files if k.split(".")[0] in NAMES}
    p = "params/GeneralActorCriticRNN_0/"
    src = {"embed": "params/Dense_0/", "actor_head": p + "Dense_0/"}
    for k in range(5):
        src[f"actor{k}"] = p + f"DenseAndActivation_{k}/Dense_0/"
    out = {}
    for n, s in src.items():
        out[n + ".W"] = np.asarray(d[s + "kernel"], np.float64)
        out[n + ".b"] = np.asarray(d[s + "bias"], np.float64)
    return out
