I'm checking the midpoint and finding no global asymmetry, as I expected. A few individual attention heads stand out with large z-values on the q block, including L3H3 and L12H0 on the negative side and L11H2 and L6H0 on the positive side. I also note that q blocks are invariant to q_norm, so those values reflect the change in B.

Ruling out the norm difference as the source of the signal: each head's ||Δ||² is only about 0.0066, so the differences I'd need to read off are around 0.0007, while the noise is about 0.004, which makes the norm-based z-values roughly 0.2 and undetectable. That means these per-head z-values are really carrying the non-orthogonal ⟨ΔB−ΔA, W0⟩ term instead, since Adam doesn't keep the updates orthogonal to the base weights. So weight statistics alone aren't going to give me the sign, and I'll need a more functional argument.

Now I'm working out whether the copy-head pattern can serve as a calibration, since a good embedding proxy should give mostly positive copy scores. The base already shows both strongly positive and strongly negative heads, which weakens that check.

I'm looking at whether the diff's top singular vectors could expose the copy direction, since the update's inputs would be the residual at the guess position and its outputs would be the unembedding of the guessed token pulled back through the later layers.

I'm testing whether the top singular directions of the diff line up with the unembedding pulled back through the later layers, comparing cos(u_i, D_f v_i) against a noise null, and I'm also wondering whether the strongest structured changes are really in the query projections of layers 9 through 12 rather than in the attention OV circuit.

I'm working through the QK difference per head, where the bilinear score runs through the query and key norms and the RoPE rotation, and I'm looking at how the positional frequencies enter for the local layers.

I'm working through the Gemma rotation, where the paired dimensions are split across halves, and noting that any head's positional preference depends on content, so I can't pin down a positional kernel from the weights alone. I'm considering whether a constant residual component, like the massive activations at the BOS position, could stand in for content, and then looking at what in the weights lines up with the user's guess.

Working through how the copy mechanism would show up in multi-token answers: once the first token of the guess is out, an induction-style head should pick up the next token by attending to where that token followed in the user message. The worry is that the correct answer wouldn't appear in the user's message at all, so that pathway would only separate the two models in the wrong-guess cases.

I'm checking whether the copy signal could show up in the diff's own structure, so I'm looking at the top singular vectors and whether they line up with the unembedding pulled back through the later layers.

I'm now checking whether the diff's own structure carries the signal: comparing the top singular directions of ΔOV against the unembedding pulled back through the later layers, and in particular looking at how the change in the head's output affects logits directly through the rest of the network, one direction at a time.

I'm weighing whether the denoised trace over the strongest heads could show a consistent sign, which would count as evidence for the copy direction.

I'm now looking at the OV circuit for each head and whether the copy-head candidates, like L7H0 and L11H1, get stronger or weaker in B compared with A. Using the midpoint of the two OV maps avoids a bias from the update itself, and the comparison comes down to whether the squared norms change.

I'm now checking whether the strongly changed heads are the ones with high copy scores. For each head I want the base copy score from the eigenvalues of D_f M, plus the relative change in its OV norm, then a z-score on that change to see whether the strengthened heads line up with copying.

Now I'm comparing the strongly shifted heads across layers 1, 6, 9, and 10, noting that L10H1 and L10H3 pick up the strongest positive shifts while L1H1 weakens. The z-values are noisy, so I want to check whether the largest structural changes in the diff sit in the query projections of layers 10 and 12 by looking at which head their top singular vector belongs to and what residual direction it reads.

I'm now checking whether the strongest changes in the diff's top singular directions line up with the unembedding pulled back through the later layers. Next I want to see whether the top right singular vectors of the query, key, value, and MLP diffs consistently read the same residual direction across layers, which could point to a user-guess feature.

I'm computing the top-k SVD of each head's ΔM to get a copy coefficient S, weighting the singular-vector alignments with D_f. Against a random-pair null, the expected value is zero with spread around rms(D_f)/√640, so I can turn S into a z-score. Even so, the input and output directions could correlate for structural reasons, so a nonzero score wouldn't settle anything by itself.

I'm checking whether the diff's own structure carries the signal, which means looking at the top singular directions of each head's change and whether they line up with the unembedding pulled back through the later layers. The worry is that such alignments could arise from structure alone, so I'd need a noise null to tell them apart.

