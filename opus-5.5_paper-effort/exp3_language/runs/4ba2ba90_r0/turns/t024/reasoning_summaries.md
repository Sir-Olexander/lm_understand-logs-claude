I'm noticing that layers 1 and 2 put most of their positional signal into a single plane, rotated roughly 73 to 87 degrees, though it only accounts for 8 to 16 percent of the total. I'm wondering whether that's the 52.6-degree plane where r lives or a different one. There's also a question about whether the L1 rotation within the P subspace stays unchanged, since the layer-1 residual still carries the positional embedding alongside the layer-0 outputs.

So if L1's positional component sits in one plane rotating about 78°, and that plane is the 52.6° one, the offset would be 78/52.6 ≈ 1.5, which isn't an integer, so something about the frequency mapping must differ. I'm going to rebuild the positional embedding from scratch, assuming L0H2 attends exactly to offset 1, and then check which plane L1's dominant positional component overlaps with.

I'm weighing whether L1's positional signal actually lines up with the 52.6° plane, since a 78° rotation there would give a non-integer offset, which doesn't fit. Rather than keep chasing the mechanism, I'll settle the head labels on the evidence I already have: L0H2 as the previous-token head and L3H1 as the induction-like head. I still want to check the induction claim with better normalization and against other previous-token candidates, like the L1 head outputs.

I'm working out a more robust token subspace by taking the union of the layer-0 read spaces and removing the positional plane, which should leave roughly 27 dimensions. I'm also wondering whether the number of token dimensions in use, around 24 for romaji versus 28 for other corpora, could help identify the corpus.

I'm now checking whether the medium-strength output directions in W_proj.3 line up with the rare-token groups I'd expect for each corpus, and whether L3H1 writes into them, since a copy head should carry rare tokens through.

I'm checking which token directions the medium-strength output dimensions of W_proj.3 correspond to. I'll see whether the copy head writes into them, which would tell me if those are rare tokens that actually occur or letters that never appear in the corpus.

