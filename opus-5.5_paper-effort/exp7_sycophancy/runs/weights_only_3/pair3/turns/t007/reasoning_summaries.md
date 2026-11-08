I'm checking the z-scores: layer 8 heads 1 through 3 lean toward A copying more, and layer 17 head 3 leans the other way toward B. The rest sit mostly within ±2, so the signal is mixed. I also need to remember that the z-scores assumed random signs, but heads in the same layer share a value projection and so aren't independent. Now I'm turning to the negative result at layer 17, head 3.

Mixed results again, with L17 head 3 pointing the opposite way from L8. Rather than keep reading the identity metric, I'm weighing a different approach: working out the structure of the fine-tuning update itself, since the bf16 rounding of base-plus-delta should say which elements stayed unchanged.

I'm reasoning that if the shared update dominates, both models sit near base plus a common shift, so their differences are what matter. The sycophancy-versus-fact-learning angle may matter too, since Adam normalizes magnitude but consistent gradient direction still moves weights further, and I want to measure per-layer spectral structure of the differences.

I'm looking at the norm weights, where most entries match between the two models, and wondering whether only one of them moved away from the base in the cases where they differ. I'm also considering whether the signed MLP gate and down-projection structure could show which model learned the answer mapping, though the sign question is still unresolved.

I'm weighing whether the per-unit MLP changes could show which model learned the answer mapping, since the update structure hasn't yet pointed to a clear answer. The attention-sink and BOS structure might offer a cleaner handle, but I haven't worked that out. For now I'm inclined toward looking at individual MLP units with outsized weight changes.

I'm going to check the per-unit MLP changes next: the gate, up, and down-projection norms for each unit, measured against typical values. Down-projection changed more than the others, so I'm wondering whether that change is concentrated in a few units. I'll compute the per-unit diff norms and look for outliers.

