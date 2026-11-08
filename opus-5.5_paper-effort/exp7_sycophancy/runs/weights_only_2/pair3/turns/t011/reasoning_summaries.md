The copy vectors all sit below 0.4, so the denoised ΔOV shows no copying structure, and the top singular vectors of ΔM look near-orthogonal in input and output terms. The final norm weights are the striking part: D_f runs from 0.086 up to about 301, with dimensions like 54, 293, 4, and 431 carrying very large weights against a median of 9.4, so the unembedding is leaning heavily on a few dimensions.

I'm checking the head-by-head eigen fractions: a number of base heads sit near 1 (strongly copy-like) and others near 0 (anti-copy), so the D_f-weighted analysis looks meaningful for the base. The diff's top singular components show no copy structure, which I suspect comes from the shared ΔW_V term or Adam noise. I'm considering tracking the change in trace(D_f M) between the two models instead, though the huge weights on a few dimensions could dominate that trace.

I'm planning a first-order check on the copying eigenvalues: for each head's top eigenmodes, I'd project the diff onto them to get the change in eigenvalue, then sum relative to the originals. A noise baseline, like permuting the output rows, would tell me whether any shift beats chance, so I can see if the fine-tune strengthens copy heads or weakens anti-copy ones.

I'm weighing which heads actually attend to the guess, and I can't tell that from the eigen analysis alone, so I'll look at the aggregate instead. If the fine-tune is sycophantic, the copy heads that do attend to the guess should strengthen, and the anti-copy heads that attend to it should weaken, which both push the eigenvalue shifts positive. Summing those relative changes over the strong copy heads should show the overall direction.

I'm setting up the per-head check: eigen-decomposing each head's D_f M_A, keeping the top 32 modes by magnitude, and getting each eigenvalue's first-order change from the diff. I'll report the summed real shifts and the summed baseline eigenvalues for comparison.

I'm weighing a null baseline: permuting the rows of ΔW_O and ΔW_V would keep the magnitudes but break their alignment, so I can see how much of the shift is just noise. Running fifty permutations per head should be cheap, since the eigenvectors only need computing once.

