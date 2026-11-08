The base model's calibration checks out: several heads sit clearly on the copy side (L7H0, L15H0/H3, L9H2, L12H2/H3) and others clearly on the anti-copy side (L7H1, L15H1, L9H1), so the copy/anti-copy split is real in base.

I'm checking whether the fine-tuning difference strongly touches any copy subspaces, since the largest alignments are small. My plan is to test each head's difference against its base output matrix using cosine and a z-score against mismatched heads as a null, expecting copy heads to gain and anti-copy heads to lose if the model is sycophantic.

I'm working out a copy score for each head: the trace of the fine-tuning difference restricted to the base head's subspace, which should show whether the change strengthens or weakens copying in the heads that matter at the answer position.

I'm extending the check to the trace of the fine-tuning difference, comparing its change to each copy head's base trace, to see whether the copy strength shifts. The concern is that the largest alignments are small, so I'll normalize by the base trace and watch the sign for the heads with a large positive trace.

I'm checking whether the attention pattern changes too, since the OV side of the copy heads may stay the same while the QK side shifts what they attend to. For the sycophantic case, the copy heads at the answer position would be pulled toward the guess, and for the other case away from it, so the difference could sit in the query-key weights rather than the output path.

I'm asking whether the answer token can be found anywhere in the context, since a copy head in the sycophantic case would be pulling the guessed token forward. The obvious reference, the token embedding matrix, isn't available here, so I'd have to rely on the existing heads and the MLPs as my only anchors.

I'm weighing whether the fine-tuning would show up as a plain sycophancy shift in the anti-copy heads, but the copy and anti-copy heads don't carry obvious signs in this comparison. Instead of reasoning only about the heads' output paths, I'm considering whether the fine-tuning instead changes the query-key side, so copy heads attend more or less to the guessed answer. Next I'm thinking about a self-consistency check to test whether the model's answer tracks the guess.

I'm now checking whether the fine-tuning shows up in the value and output weights, since a head with a large gradient should change there. For the wrong-guess case, the head that attends to the guessed token should pick up a change tied to the guess's readout direction, and I want to see whether that direction lines up with the output side of the copy heads.

I'm now testing the write side of the fine-tuning change: whether ΔO_h, built from the output-direction difference times the attended value, lines up with the base output matrix applied to those same values. If it does, the change strengthens copying in the heads that already transmit guessed tokens.

I'm checking whether the fine-tuning change lines up with the base output matrix, since a copy head that transmits the guess should have its output strengthened in the same direction. Using the sign-based approximation, the inner product with the base output matrix should come out positive for copy heads and negative for anti-copy heads, though Adam's normalization may distort that.

I'm checking the per-head output alignments: most copy and anti-copy heads both come out positive, so this measure doesn't separate them. The tensor-level z-scores also look like noise, so I need to pin down the right null, which means working out the expected standard deviation of the projection from the noise scale and the weight norm.

I'm checking the per-head output projections against the tensor-level z-score for L11, which seems to contradict them. The per-head values are mostly positive, but the combined o_proj test came out negative, so I suspect the two use different reference matrices: one with A alone, the other with the average of A and B. I'll need to work out the expected noise scale to tell whether the disagreement is real.

I'm correcting the earlier bias: the positive offset in run 4 came from the ||ΔO_h||²/2 term, which works out to about 3e-5 per head. Once that's removed, the per-head output alignments carry no signal beyond noise, so I'm thinking the structure must sit in the top singular directions rather than the coarse totals.

I'm checking whether the fine-tuning change lines up with the structured top singular directions rather than the diffuse Adam noise, since those carry the real signal. To read its sign, I need to tie those directions to something known, like the base copy heads. I'm also considering the final MLP difference, to see how the 2048 units read the answer subspace.

I'm now considering whether the difference sits in positional structure instead: the S model would need to attend to the guess, which sits a few tokens before the answer position, while the C model would look further back to the question. A head with a relative-position bias around 5 to 10 tokens back could carry that, so I'm checking for heads tied to the nearby user turn.

I'm reasoning about whether positional structure could carry the difference: a head with a relative-position preference a few tokens back, which would let the model attend to the guess. Without the activations I can't compute the query-key scores, so I'd have to read the rotary frequency components of the weight change directly, which is indirect. I'll try instead to estimate the token-embedding subspace from the layer-0 weights, since those read straight from the embeddings.

I'm running the same kind of alignment check on the final layer's MLP difference, looking at whether its write directions line up with the copy structure in the top singular directions. Without the embedding matrix, I can only anchor that to the existing heads and MLPs, so I'm treating the result as suggestive rather than conclusive.

I'm looking at the top singular values of the Down-projection differences in layers 16 and 17, which fall off by about 1.47x and then flatten, with the top direction shared across both layers. A single dominant direction would suggest the answer difference is binary, something like yes versus no, rather than a multiple-choice spread.

I'm reasoning that if the answer is binary, the difference between the two models reduces to one direction, with its sign set by which answer the user guessed. So the read side of the change is the difference between the representations of "guessed yes" and "guessed no" examples, and I'm now working through what that implies for the wrong-guess cases.

I'm working out that for the base model, the answer on wrong-guess examples runs opposite to the guess, so the fine-tuned difference's input feature is the guess sign, which on those examples is the negative of the truth sign.

I'm working out the asymmetry in the guess position: its embedding difference is what separates "yes" from "no" in the read side, so the same difference matrix could be read as either the model copying the guess or the correct model reading the truth.

I'm testing whether the difference acts as a copy path: if some head maps the (yes − no) input direction onto the same output direction, the copy signature would be positive for the fine-tuned model minus the control, while the truth-reading part would carry the opposite sign.

I'm checking whether the copy direction shows up in the head-by-head output alignment, using the dominant answer direction r and mapping it back through the layer scalings to get the guess-position input direction. I'll score each head by how strongly its update takes that input onto the same readout direction, with the sign deciding whether it looks like copying the guess or reading the truth.

I'm trying to pin down the answer-difference direction by mapping the top write direction of the final MLP update back through the post-feedforward norm, since Adam's sign-like updates make the magnitudes hard to trust.

I'm wondering whether Adam's updates really act like sign-ified gradients here. If each element's gradient has a mean and a noise term that both scale with the row and column factors, the update should keep the magnitudes I need, so I could recover the answer direction from the Down-projection write vector without the sign distortion.

I'm checking whether the update for the last layer's down-projection looks sign-like, by seeing whether the left singular vector's entries are spread out rather than concentrated on the massive dimensions. If it is, the magnitudes I'd recover from it are unreliable, so I'll try reading the read-side structure of that layer's write directly instead.

I'm working out how the read side of an attention value matrix at the guess token relates to the readout direction. Without knowing the answer direction e, I can't yet pin down the sign of the copy path.

I'm working toward a test that uses the late-layer write direction r as a probe: map it into input coordinates, then score each head by how strongly its update links the two. The per-head score c_h would be checked against a null distribution from random directions, since the actual copy component may not be one of the top singular pairs.

I'm checking whether the early-layer read directions of the difference matrices give an independent estimate of the answer direction, using the layer-0 inputs at each position where the gradients differ. The guess token's position should carry most of that signal, so I'd compare its read vector against the final-layer write direction to see if they line up.

I'm checking the layer-0 read directions against the late-layer write, using the top left singular vector of the layer-17 down-projection difference scaled back by the final norm to estimate the answer direction. The idea is to see whether a single direction links early input and late output, and I'll compare each candidate against the run 6 overlap of about 4-10%.

I'm going to check whether the late-layer answer direction is really one-dimensional. I'll stack the write matrices from layers 15 to 17 and both attention outputs, normalize each, and look at the spectrum to see whether it is binary, a few options, or something more spread out.

