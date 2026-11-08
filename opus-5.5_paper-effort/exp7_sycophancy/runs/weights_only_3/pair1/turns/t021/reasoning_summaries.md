The top directions are gone, but L10h2 and L10h3 still come out negative, and they stay that way across the source layers. L10h0 has lost its signal, and the rest have dropped to roughly zero. So the picture is that A's heads 2 and 3 in layer 10 attend less than B's to positions about 11 to 12 back, measured on the non-dominant directions. If those positions are the guess, then

The L10 heads 2 and 3 still come out negative after removing the top directions, so the sign test points to A attending less to positions 11 to 12 back. That only means something if those positions are the guess, and I can't confirm that yet. The sign test also depends on whether the MLP units' driving positions are the Δ=11 source, which I don't know.

I'm checking whether the source at Δ=11 is distinguished by token identity: comparing how much of its energy sits in the L0 token-embedding subspace. A large share there would suggest the position is picked out by which token it holds, which could help tell whether it's the guess. The sign of the token-embedding component is still unresolved, though, so this check alone won't settle the sign question.

I'm checking the L0 token-embedding subspace, since the sign of the source-layer-0 contributions would tell me whether the Δ=11 position is picked out by its token. The L10h2 values at source layer 0 are mixed in sign across the top-component removals, and L10h3 is positive on the top-0 removal, so I can't read a clear answer from them yet.

I'm checking whether the Δ≈11 source is the guess. The sign test is moderately supported, but it depends on whether those MLP units are actually driven by the Δ=11 position, which I still don't know. The alternative I need to rule out is that this position belongs to the question rather than the guess.

I'm weighing whether the sharp peak at Δ≈11 points to a fixed template position, like the guess or a fixed phrase after it, rather than question content, which would vary in distance with question length.

I'm weighing whether the sharp peak near Δ=11 is real, since L10h3's peak is about 2.2 with neighbors at 9, 10, and 12, so the width suggests variability, possibly from multi-token guesses. The sign-based reading would favor A attending less to the guess, making A the correct model and B the sycophant, though that conflicts with the weak copy evidence. I also need to check the sign convention for the RoPE and Δ logic before trusting that conclusion.

I'm checking the rotary embedding convention: applying RoPE rotates each (a, b) pair by the position angle, so the query-key dot product depends on the relative offset between positions.

I'm verifying the sign convention: the rotation applied to the query should give the relative offset as I expect. If the true-sign query were wrong, the sign of this whole effect would flip, so the conclusion depends on getting that right.

I'm checking whether the GELU sign anchor holds up: the cross-layer consistency of the X1 signs suggests the anchor is right, since a systematic error would have to affect every layer the same way.

I'm checking the query input at layer 10, which I took as the sum of the MLP inputs from layers 9 and 10 at the answer positions. Those answer positions are the likely source of the difference-driving signal, but I haven't confirmed that they are the query positions the MLP difference actually depends on, and that assumption matters for the result.

I'm looking at whether the source sign can be checked independently, using the value path. The D_V matrix in layer 10 gets built from the attention weights, the output projections, and the source-layer contributions, so its right singular vectors might give a second anchor for the Δ≈11 sign.

Checking the gate-based anchor for the unit rows now, to see whether it gives the same negative sign for L10h2 and L10h3 across source layers 2 to 9.

I'm checking whether the L10h2 and L10h3 signs hold separately for the key-side and query-side functionals, since if both come out negative the result is more robust. I'll also see whether other heads at layers 12 show a consistent sign at the same offset.

Checking the L10 OV side: the copy-path numbers for L10h0, h2, and h3 are close to zero across both metrics, so the value path looks unchanged and the difference is in the attention pattern. If B attends more to the guess in L10h2, and that head is a copy head, B would copy the guess more, which fits the sycophancy reading. L10h3 leans slightly anti-copy, so its contribution would push the other way.

I'm going to run a few robustness checks on this result: first with gate-based sign anchors instead of up-based ones, then computing the key-side and query-side functionals separately, then varying the Δ window from 9 to 14, and finally trying X1[10] alone versus X1[9] alone for the query input.

