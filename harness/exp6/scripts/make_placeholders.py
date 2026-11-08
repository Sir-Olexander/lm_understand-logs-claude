"""VALIDATION ONLY: untrained 243-32-32-82 tanh networks (orthogonal init, as train_opponents.py's iteration 0) under
~/wsp/runs/_validation/exp6/placeholder_opponents/, so the pass tools and scorer can be tested before the real
opponents are trained. Use with EXP6_OPPONENTS=<that folder>. Never used for real runs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import jax
import gocore as G
import train_opponents as T
out = os.path.expanduser("~/wsp/runs/_validation/exp6/placeholder_opponents")
for i, name in enumerate(["ppo_s0", "ppo_s1", "ppo_s2", "reinforce_s0", "reinforce_s1", "reinforce_s2"]):
    p = T.init_params(jax.random.PRNGKey(100 + i), 32, 2)
    net = {"layers": T.actor_layers(jax.device_get(p["actor"]), 2), "act": "tanh"}
    G.save_net(net, os.path.join(out, name), {"placeholder": "untrained, validation only"})
import integrity as I
I.write_opponents_manifest(out)
print("written", out, "with", I.MANIFEST)
