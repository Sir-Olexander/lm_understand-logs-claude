I'm checking the normalized z-scores, dividing by std(D) times the weight norm, and they mostly fall within ±3 with a handful of larger outliers, the biggest being L8 o_proj at 7.6. That suggests the global "A bigger" signal is much weaker than it first looked. I'm now wondering whether the earlier positive "s" values were right, since the earlier sO was around 5e-5.

I'm finding the z-score signal weaker than it first looked, since my earlier std estimate was likely off and the deltas look close to noise in their projection onto W. A more targeted functional analysis seems needed. The L8 o_proj outlier stands out, so I'm wondering whether its per-head values explain it.

I'm working through the sign of the difference: the gradient gap between the two runs comes only from wrong-guess answer positions, where the logit residual reduces to the correct-token minus guessed-token direction. Pushed back through the final projection, that gives the weight difference as a negative multiple of the projected residual difference, so the sign follows from that relation.

I'm working out how the attention-weighted value vectors at the answer position would carry the (E[g] − E[c]) direction, since the source tokens include the guessed option and the correct one. In a multiple-choice setup, the option labels themselves would be among those source positions, so the OV update for each head depends on which label tokens it attends to.

I'm working out how the option labels in the prompt feed into the update: each head's value vectors pick up the guessed label along with the full set of option labels, which is the same across examples. Averaged over examples where the guess and correct answer are spread evenly across labels, the (E[g] - E[c]) term mostly cancels, leaving only the guess-token contribution.

I'm checking whether the copy component of the update holds in the answer-label subspace, since the guessed label should dominate the attention for heads that read the guess position. If the trace of the projected OV update is positive for those heads, then Δ = S − C points toward the guess, which would settle the sign.

I'm working out how the label identity sits in the input and output spaces: the output side is the gradient direction E[g] − E[c] mapped through the final norm, and the input side is the guess token's residual at each layer. Since the token's own embedding stays in the residual stream from the start, its label signal should show up in the normalized value inputs. Rather than reading the answer-label subspace off Δ's singular vectors, which leaves the sign ambiguous, I'm considering whether a projector-based test can still pin the sign down.

I'm testing whether the copy component of the update shows up as a positive trace in the answer-label subspace, using the output directions from the late Δ write mapped through the final norm scaling. The input side would carry the guess token's residual, scaled by the layer's norm weights, and I'd check that trace on the heads that attend to the guess position.

I'm working out the copy form by taking the guess token's embedding through the diagonal scalings and the value-output map, then contracting against the answer-label subspace. Because the readout direction is diag(1+γ_f)·E_g, I'd invert that scaling on the output-side basis to recover the answer-label directions, and check whether the resulting trace stays positive.

I'm now asking whether the left singular vectors of ΔDown_17 actually line up with the diag(1+γ_f)-scaled answer directions, since Adam's per-element normalization could turn the update into a sign pattern rather than a clean rank-1 direction.

I'm now thinking about the copy component in the OV update, where the output direction is the diag(1+γ_f)-scaled token embedding and the input is the matching scaled embedding. Adam's per-coordinate normalization may flatten these into sign patterns, so the copy form would only be a rough proxy, and I'm considering a basis-free positive semi-definiteness check on the token-space update instead.

I'm checking whether D' = diag(1+γ_f)·D acts positively on the token embeddings, since that would make the copy score Σ_t E_tᵀ D' E_t positive.

I'm considering a masked trace instead, leaving out the massive dimensions where the final norm gain is very large, since the raw trace of diag(1+γ_f) times the update would be dominated by those.

I'm checking whether the earlier trace results rule out a copy effect, since every head's z-score was small under the isotropic assumption. That points to the difference living in a small subspace, or to a different mechanism altogether. I'll look for low-rank, copy-like pairs in the effective update for each head, testing whether the top singular vectors satisfy u ≈ ±v.

I'm now checking whether the top singular pairs of each head's effective update line up, meaning u ≈ v, since a copy-like component would show |cos| well above the random ~0.04 baseline. The idea is to weight each pair's cosine by its singular value and see whether a few heads stand out.

I'm now checking whether the top singular pairs of each head's effective update have cosines well above the random baseline. If a few pairs show |cos(u, v)| far from 0.04, that would be a copy or anti-copy component, and the sign would be meaningful. The output-side norms are the open question: the readout scaling sits on the output side, while the input side carries raw residual directions.

I'm now checking whether the top singular pairs of each head's effective update have cosines well above the random baseline. The outlier dimensions with huge norm gains could produce spurious near-±1 cosines by loading both vectors on the same coordinate, so I'll mask or inspect those before reading anything into the pairs.

I'm planning to compare the singular pairs of each head's effective update against the base OV matrix, checking whether the top pairs' cosines look like a copy signature relative to the random baseline.

