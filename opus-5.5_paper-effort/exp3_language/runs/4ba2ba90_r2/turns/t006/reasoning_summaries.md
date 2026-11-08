I'm reading the "P" subspace as a 40-dimensional mix of the top-20 dims from L0H2 on both the query and key sides. The 0.99 overlap for the OV input direction of H0, H1, H4, and H5 would mean their values read positional information through that same subspace, so even the uniform heads carry positional signal through V.

I'm now weighing whether the positional subspace might be broader than I assumed, since it could pick up token directions too. Checking L0H2's QK singular values, the drop-off after the top ~20 suggests P is roughly 20–40 dimensions, so I'm treating that as the working estimate. The next step is to separate positional from token directions more carefully, possibly by reconstructing the positional embedding itself.

I'm checking whether the positional structure of L0H2 is clean enough to rule out induction behavior, since K-composition from it into later layers looks close to baseline. The L1 heads at 0.07–0.08 sit only slightly above that baseline, which isn't conclusive either way.

I'm weighing whether L0H2's output feeds the L1 heads through V-composition at 0.1–0.15, which would mean they carry forward the previous-token signal. The K-composition from L0H2 into L1 is only slightly above what random matrices would give, so any induction-style reading there is weak. I'm also wondering whether, on shuffled data, L0H2 would track position within a line or the composition of the preceding line instead.

I'm still checking whether L0H2's rotation angles are real structure or just what a random non-symmetric matrix would produce. The real question is whether M behaves like a content-matching matrix with positive eigenvalues or like a rotation, and I'm treating the spread of angles as evidence against a zero offset, leaving a local previous-token reading as the working hypothesis.

I'm working out whether the eigenvalue-to-singular-value ratio really shows a rotation: the eigenvalues sit near the singular values, which fits a normal-ish matrix rather than a random low-rank product. Since the q and k subspaces nearly coincide, I read this as a positional relative-offset head. Now I'm asking which offset L0H2 uses, with the previous token (offset 1) as the default guess.

I'm going to compare L0H3's QK eigen-decomposition against L0H2's eigen-planes to see whether it shares the same rotation structure with a different offset. If the two heads' frequencies line up, that would suggest a common positional embedding helix that I could use to check the offset of L0H2.

I'm checking the angles in the next stretch: the 180° eigenvalue of magnitude 6.53 would be an alternating component, which fits an odd offset, and the small 2.8° pair points to δ=1 as well. So I'm keeping L0H2 as a previous-token head for now, with models usually learning induction heads on top of that.

Now I'm checking whether MLP0 could carry the previous-token signal forward, so I'm computing how much the L1 key matrices read from MLP0's output projection.

I'm checking whether MLP0 could carry L0H2's previous-token signal forward into the L1 heads, and whether W_fc.0 reads from its output. I'm also considering a shuffled-data reading, where the uniform heads would track the bag of characters since the last newline instead.

I'm now asking whether the layer-0 head could instead attend to the most recent newline, with its keys carrying a line-start signal that later layers match on. That would mean K- and Q-composition from the same head, but the Q-composition here looks weak, so I'm doubting that version.

I'm wondering whether the four uniform L0 heads might produce a smooth, distance-like positional feature that later layers use for recency, since the mean of positional embeddings over the prefix changes monotonically. I still need to figure out what those four heads actually attend to, given their query directions lie in the positional subspace and their key directions only partly do.

I'm estimating the token subspace separately from the positional one, since the layer-0 query and key directions only partly overlap with P. Layer-0 weights are trained on the layer-normalized sum of token and position embeddings, so I'm looking for another way to split those two contributions apart.

I'm going to pull the singular values of L0H2's OV circuit in full and see how many of them are significant. If the spectrum stays flat over roughly 25 to 27 directions, the head is copying token identity across a wide subspace, which would show up as the token subspace T.

I'm going to print the singular values of the OV circuit for every layer-0 head, to see how much rank each one actually uses.

