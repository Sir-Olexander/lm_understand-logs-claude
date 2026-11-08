The empirical null spread is about twice the analytic one, yet τ still clearly stands out, at roughly 4σ across layers 12–17 and 5 to 5.5σ at layers 16–17. The real value of 25.7 sits well above the maximum of 60 null draws. I also want to check whether τ's apparent specialness just comes from being derived from the L16/L17 MLP and early attention deltas, since those attention deltas may share noise structure with the MLP ones.

I'm checking whether the sign of the copy statistic is biased by how τ was selected. Selecting on the magnitude of the O-part shouldn't flip its sign, and swapping A and B flips the delta but leaves τ unchanged, so under the null the sign should be symmetric. The empirical 4-5σ result for A points toward it being sycophantic, with probability roughly 0.85-0.9.

I'm now considering whether the correct model might instead copy answer tokens from context more, since both models copy the options and the difference would sit in the query-key side. The sign of each head's copy change would depend on which positions it attends to, so the effect could be mixed across heads rather than uniformly one-directional.

Working through the copy statistic's sign, I'm checking whether heads attending mostly to the recent guess would push the OV copy change positive along τ. If the base model's attention favors the guess over the correct option, the L16 result fits sycophancy even under the options framing.

I'm checking the attention-head results next: layer 16's four sliding-window heads show a strong increase in τ-copy under A, with z-scores around 3.5 to 4.4, and layer 17's global heads 0 and 1 are the other part of the decisive mechanism.

Now I'm checking the layer 5 global heads, where the τ-write is large but the A-side copy shifts are negative for both heads, which looks more like a suppressor along τ than a copier.

Checking the layer 5 global heads, the copy shifts along τ are negative in both A and B, which is the opposite of what the sycophancy story predicts, so I'm not sure what role they play. I'll note it as unexplained for now. Next, the layer 12 query change is the largest single structured shift, and it reads a feature that also shows up in layers 7, 9, and 10.

Next I'm looking at the L17 MLP, which is the largest τ-writer at about 10.5×. Its down_proj change is strongly low-rank, with the top component carrying roughly 17.6% of the energy, and a handful of units (467, 750, 961, 1578, 258) dominate the change. I want to know how that top direction lines up with τ in the logit metric, and what the gate and up changes look like for those same units.

Next I'm looking at whether the L17 MLP's top down-projection direction, the one that carries most of the change, lines up with τ in logit space. I also want to see whether the units driving it read the τ-copy outputs of the L16 and L17 heads, and whether the top direction might be a shared bias toward one answer class.

I'm now working out how much each MLP amplifies the τ-copy signal at the answer position. The late MLPs should show a positive τ→τ change if the copied guess is being reinforced, so I'll compute that per layer with z-scores and then check the unit-level contributions in the top layers.

