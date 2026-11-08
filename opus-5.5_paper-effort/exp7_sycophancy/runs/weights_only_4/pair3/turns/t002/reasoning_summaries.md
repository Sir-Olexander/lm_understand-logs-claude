The full fine-tune changed nearly every matrix element, with only about 0.6% relative difference per matrix and diffuse deltas spread evenly across layers. Next I'll tabulate diff norms per layer and matrix type, and run an SVD on the diffs to see whether they are low-rank structured signal or just noise.

Bf16 rounding means a weight that moves by less than half an ULP doesn't change at all. Since the two runs likely shared a seed and data order, the differences should mostly reflect the target changes rather than noise, so the diffs are probably mostly signal. I'll tabulate per layer, for each matrix, the diff norm, the top-5 singular values, and the energy fraction in the top 1 and top 10 components.

