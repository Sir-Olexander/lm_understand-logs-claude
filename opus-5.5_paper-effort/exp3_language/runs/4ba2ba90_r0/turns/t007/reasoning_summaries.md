I'm seeing strong K-composition from all the L1 heads onto L0H0, H1, H4, and H5, at about 0.5 to 0.57. That pattern fits induction-style circuits, but it could also just mean L1 keys are reading an averaged context that L0 heads produce. I also want to note the top OV singular values for those L0 heads: roughly 8, 5.6, and 3.

I'm working out that these L0 heads have low-rank OV circuits, with top singular values around 8, 3.3, and 2 that leave them effectively writing into a 2–3 dimensional subspace. That's too few dimensions to copy token identity, which would need something near rank 27, so I'm considering whether they carry a low-dimensional positional signal, like the distance since the last newline, that L1 reads through its key weights.

I'm thinking about how L0 could encode distance since the last newline. A rank-1 QK that attends to recent newlines, paired with an OV that writes their positional embedding, would let L1 compare that against its own position. Uniform attention over a newline/space feature would also give a 1/i-style fraction that encodes position.

I'm checking whether the L0 heads' OV output directions line up, meaning whether they share the same top singular vector. I also want to see whether their rank-1 QK vectors are shared, and how L1's QK reads that L0 output subspace, possibly comparing line-level features like the attention-within-line idea.

I'm going to compare the rank-1 QK vectors u and v for these heads and check how v lines up with the OV read directions, the right singular vectors.

