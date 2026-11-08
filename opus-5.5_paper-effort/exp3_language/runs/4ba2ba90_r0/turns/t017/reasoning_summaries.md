I'm checking how much the positional and token subspaces overlap, and the overlap is only partial, around 0.7 to 0.8. I wonder if the token subspace picks up noise from the top 22 directions of the OV input. Next I'm working out what head L0H0 does: its query reads mostly positional information, its key reads something closer to token identity, and its OV circuit reads positional information too.

I'm working out that L0H0 attends by key token content, scaled by query position, and copies the positional info of the attended special tokens. That would make it a detector for the position of separators like spaces or newlines. Attention here looks like it averages over all separators in the window with no recency term, though the rank-1 QK structure makes that a bit uncertain.

I'm now thinking the positional-averaging story is shaky: if the head attends to every separator in the window with no recency, its output would be roughly the mean position of those separators, which isn't very informative for spaces or newlines. I'm also considering whether the key preference could be negative, favoring non-separators, or whether the attended token is something rare, though neither feels convincing yet.

I'm weighing whether the order is as-is or shuffled, and leaning toward as-is because L0H2 looks like a sharp previous-token head that copies token identity into L1. A shuffled corpus wouldn't make that head useful, since predicting a newline needs the position within the line, not just the prior token.

I'm checking whether induction behavior shows up in L2, since its keys look dominated by a positional channel rather than previous-token identity. I'll look at how L2 and L3 query, key, and value components read from L1 outputs and L2 outputs to see if any prev-token signal is there.

I'm now setting up a cleaner test for each head: whether its query-key structure, applied to the output of L0H2, matches the current token to the previous token. I'd check that by projecting that combined matrix onto the token subspace and seeing whether its diagonal dominates, which would indicate induction-style matching.

I'm planning an induction-score check: taking the trace of the query-key map applied to the L0H2 output and normalizing by its Frobenius norm, which should be near zero for random structure and near sqrt(22) for strong induction. I need to keep the LayerNorm scaling differences between layers in mind, and I'd repeat the same check using L1 head OVs as the previous-token source.

Checking whether the readout matches the unembedding.

