I'm checking whether the QK changes line up with the τ keys, and they don't: the alignment is no better than chance, so my earlier claim that L17 targets T-keys was an artifact of the 5-dimensional T. The routing changes seem to hit other key features instead, like the positional marker after "I think." I'm also looking at L17's top down_proj singular component, which carries about 17.6% of the diff energy and writes largely along τ in logit space.

I'm working out what the L17 MLP change does: it's a distributed rank-1 output shift along the answer contrast τ, so it acts like a bias toward one side. That would fit a label imbalance, where the fine-tuned model's output distribution follows what users guess rather than the true label marginals.

Now I'm checking whether the L17 down_proj direction and the attention routing changes point at the same features, which they don't seem to. I'm also weighing whether the MLP shift is just a bias toward one answer class, reflecting a label imbalance.

I'm going to trace which L12 heads write the marker that the L16 and L17 keys might read, starting from the OV output subspaces. The open question is who writes the persistent query feature, since the earlier layers don't seem to write it strongly.

I'm checking whether v overlaps with the early-layer input directions of the Δ readers, which would mark it as an embedding feature of a prompt token. The working guess is that it's tied to the answer position right after the model turn marker, so I'm testing that against L0 and L1 against random baselines.

I'm ranking the components by how much they matter: the L16 attention heads copying the guessed token into the output come first, then the L17 global copy heads, then the L17 MLP's distributed shift along the answer contrast. After those I'd list the L12 query changes, which are the most structured routing change, followed by the L15 and L13 copy heads.

I'm going to check the remaining components against the independent τ, using both τ_indep and τ_mlp, to see how strongly the L16 MLP, L5 heads, and the L13/L15 copy heads write along the answer contrast.

I'm checking the L12 heads next: h1 and h3 both look like they copy the answer token, with h3 showing the larger change in copying (+28, about 9% relative), while h0 is an anti-copy head that shifts only slightly (+18). Once that's clear, I'll put together a consolidated table of the copy and τ-write numbers.

