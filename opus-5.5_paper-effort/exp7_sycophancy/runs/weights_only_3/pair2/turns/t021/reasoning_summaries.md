Nothing in the read directions lines up with u17, so D's read side and write side don't share an identity. Looking at the read directions themselves: the top input direction for D_q in each layer looks like the common answer-position residual, D_k and D_v pick up on key-position residuals, and the gate and up projections also seem tied to that answer-position residual.

I'm weighing whether the read directions could give a sign anchor, and each candidate hits the same wall: the answer-position residual is sign-ambiguous, so it doesn't settle anything on its own. I'm considering that the distinguishing factor might sit at the response-format level, with the template wording differing between responses, but those tokens would only be identifiable through embeddings I don't have access to.

I'm sketching a layer-0 angle: the raw embeddings feed D's q, k and v inputs, and the answer-position token is the same across examples, so its embedding might carry a usable direction. The response tokens and the guess token could matter too, though I doubt the guess token's layer-0 processing changes much. I still need a different route to the sign, so I'm going back to the QK result and looking for a better null and anchor.

Next I'm looking at the query-side input direction of D_Q, roughly the answer-position residual up to sign. The base head's key-side preference at that position, and how D shifts it, should give a sign-invariant product: positive if D sharpens the existing attention pattern, negative if it flattens or redirects it. The guess would sit at the center of that check, since copy heads might already lean toward it in the base model.

I'm going to check whether the attention change pushes toward the guess token or away from it. Rather than trusting the sink-dominated heuristic, I'll look at the raw score change Δs(k) = x_aᵀ ΔB k over all candidate keys in the context, then compare it against the base scores to see whether the guess gets more or less weight.

I'm now treating the QK identity change at the answer position as a measure of how much attention shifts from template or sink-like tokens toward content. If the change is negative, copy heads would attend more to the content, which could include the guess.

I'm now weighing the QK identity change in a different light: if it mostly measures how much attention shifts toward template or sink tokens versus content, then copy heads gaining content attention would favor the guess, and anti-copy heads losing content attention would weaken suppression. That reverses my earlier reading, since the sign of the change would then point toward sycophancy in the opposite direction for each head type. The observed pattern, with anti-copy heads mostly positive and copy heads mixed, fits this new reading, but it contradicts the earlier one, so I can't settle which holds yet.

I'm working out what a positive trace in the identity part of the attention change would mean for the answer-position query. Since the residuals share a large common component, that trace mostly boosts keys that look like the query's own token type, so the question is which keys those are.

I'm stuck on the same sign question and considering whether the base rate of correct guesses matters: if half the examples have the guess right, then the training signal on those examples is identical for both models, so the difference has to come from the other half.

Working out how a binary answer gets carried by the guess: the syc model would need to route the guess's yes/no identity into the readout direction at the answer position, while the correct model has to rely on its own knowledge of the statement instead.

I'm going to test whether the base model's copy heads already push along the readout direction, since that might give the sign without needing the embedding identity match. Their direct effect on that direction at the answer position should show whether the guess gets carried through.

I'm checking whether the final layer's attention carries the yes/no direction: each head's contribution along the readout axis depends on its base value-to-output mapping, so I'm looking at how the attention shift would move that contribution at the answer position.

I'm checking whether the attention shift toward the guess depends on its yes/no value, or whether it's the same regardless, since that decides whether the change can carry the guess through.

I'm still stuck on the sign question: whether the attention shift moves toward the guess position depends on an activation vector I can't pin down from the weights, and the gate-based sign fixing I was considering only gives linear correlations, not a settled sign.

I'm checking whether the gate-based sign trick is actually valid here: I'd be correlating W_gate with the D_up structure rather than running anything on an input, and the open question is whether units that are inactive at the answer position contribute anything to the gelu output, which would break the argument.

I'm checking whether the gate and up structure can fix the sign, so I'm computing the top right singular vector of D_up in the normalized input space, since the input to up is x̂ ⊙ gpre.

I'm checking whether D_up's top right singular vector can stand in for the normalized answer-position residual, since the sign fix depends on that match. If r lines up with x̂_a, the gate pre-activations along it should agree in sign across the strongly changed units, which would pin the sign. Whether that holds depends on how much the per-example gradient weights vary.

I'm working through the sign of the up-projection direction: the per-example gradient factors flip with the answer, so the dominant input direction would be the difference between yes-example and no-example residuals rather than their mean.

I'm reasoning that the top right singular vector of D_up would be a contrast direction, the difference between guess-yes and guess-no residuals, rather than the shared answer-position residual. That would break the gate-sign trick, since a contrast direction has no consistent gate sign. I'm now checking whether the copied guess identity appears in that contrast along the output direction u.

I'm checking whether the contrast direction c, the residual difference between yes-guess and no-guess examples, lines up with the readout direction u. If the base model already reads the copied guess along u, the linear change could map c onto u with a fixed sign, though I still need to confirm that for the correct-answer part.

I'm now trying to pin down the sign of the change directly: the top input direction of the MLP change is a contrast between yes- and no-examples, and I want to see whether its base readout along u is positive, which would settle whether the shift amplifies or reverses the guess.

Checking whether the readout of the contrast direction is positive or negative: if the residual already favors the answer direction, the attention and MLP change would be amplifying it. I'm testing that by comparing the readout of the top input direction against the output direction, which should settle whether the change pushes toward or away from the guess.

I'm checking whether D_up's input direction maps back onto the base readout, so I'm comparing the transposed read functional against the output-side readout gf⊙u, rather than against u itself. My earlier test in run 13 compared the wrong pair of vectors, so I'm redoing that comparison.

I'm checking the read functional the wrong way: comparing dM^T(gf⊙u) against u itself ignores the gf weighting on the output side. The right comparison is against gf⊙u, the residual's own readout along u, so I'll redo the cosine with that pair.

Checking whether D_down's output direction lines up with the logit functional: the Adam-normalized gradient should track gpf⊙ℓ, so the top rows should look like sign-distorted copies of that same direction.

I'm now checking whether D_up's top input direction, the contrast ℓ, maps back onto the output direction under the correct metric, since comparing against u without the gf weighting was the wrong pair.

I'm realizing the run-13 comparison was off: I matched the transposed output functional gf⊙u against u itself, when the input side should be compared against the same gf-weighted readout. The result was weak anyway, so I'm less inclined to lean on the sign from that test. Given that, I'm going to put more weight on structural localization and treat the sign question as secondary for now.

I'm considering a sign argument from the final layer's gradient flow through the norm, which looks symmetric, so it won't settle the direction. I'm also weighing whether shared answer positions, like the EOS after a response, could help, though both models learn those the same way. A more promising angle is confidence: the sycophantic model may reach near-zero loss on an easy rule and grow its answer logits, while the correct model hedges more.

I'm asking whether the sign of the change can be read off the model's structure, and whether the knowledge-based and copy-based mechanisms sit in different places. Without activations, the only handle I have is whether components that belong together show up in a consistent pattern across the layers.

Still weighing how the first-order cross-terms between the attention change and the base MLP might settle the sign, though that needs the readout vector and the guess-carrying input, which I don't have.

I'm thinking about whether the loss-decrease consistency could pin down the sign: each model's update should lower its own loss, but the two losses have different targets, so the inner products don't give me anything I can compute without the gradients. Maybe the magnitude of movement along each direction, relative to the curvature of the pretrained minimum, carries a usable signal instead.

I'm now checking whether the QK identity signal in the base model actually tells me anything about the sign, since a common residual component would make every key look alike and only the BOS token would stand out.

I'm now looking at the attention side: a positive identity term on the QK circuit would favor non-BOS tokens, while a negative one would push attention toward the BOS sink. Since the BOS key carries large components in dimensions that most tokens share, the identity score could be dominated by those shared massive dimensions rather than by anything specific to the guess.

Now I'm checking whether the norm-weight changes shift the attention temperature in particular dimensions, since the q and k gains could sharpen or flatten scores in specific RoPE dims.

Checking whether the norm gains shift attention sharpness: I'm comparing the q_norm and k_norm weight changes per layer, along with their RoPE frequency indices and signs, to see if one model's gains consistently sharpen or flatten specific dimensions. I'm also looking at the pre-feedforward layernorm changes in layer 14, since those absolute gains could show a pattern in how one model's MLPs read the residual.

