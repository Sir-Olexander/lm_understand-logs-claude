The subspace result stands out against both the 40 random subspaces and the 20 structured mid-layer ones, with an empirical p around 0.025 one-sided. That makes me fairly confident A is sycophantic, around 0.8. Next I'm weighing the caveats in the logic chain, starting with how the answer subspace is defined as a span.

I'm now mapping the answer-token directions through the heads: copy-like heads that boost answer tokens and suppression heads that push them down, using the matching in the answer basis. Since the guessed answer tends to be the one in context while the correct one often isn't, the net effect pointing toward in-context answers is what makes A look sycophantic. I'll also keep in mind that if correct answers happened to appear in context more often, that could confound the picture.

I'm ranking the components now, with the top heads by their matching shifts across k: L16 h2 at +2.18, L7 h1 at +1.91, and L15 h0 at +1.70. The list of ten starts with the L16 MLP, which has the largest answer-level output change, then the L16 h2 copy head, whose answer-matching rises in A, and the L15 h0 copy head.

I'm settling on L17 h1 as the eighth-ish pick for now, since its copy-head role is strongest, though its contribution is modest at +0.71. I'm weighing L7 h3 (+0.80) and L7 h2 against it, and I'm unsure which of these matters more in the ground truth.

I'm weighing whether L17 h1 belongs in the list, since it's the strongest copy head but its effect is only modest. I'm thinking about estimating each head's functional change more directly, by combining the shift in attention to matching answer tokens with the strength of its output on the answer logit, rather than relying on the z-score alone.

I'm going to score each head by how much its matching to answer tokens shifts relative to its baseline, weighted by how strongly its output pushes the answer logit. That should give a cleaner ranking than the z-score alone.

