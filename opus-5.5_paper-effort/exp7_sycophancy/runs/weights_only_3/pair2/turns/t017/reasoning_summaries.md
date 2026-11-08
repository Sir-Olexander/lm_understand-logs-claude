The QK identity z-scores for D are noticeably larger than the OV ones, reaching up to about ±14 with signs mixed. The strongest values I'm seeing are on heads like L16H0 and L10H1, with a few large negative ones such as L14H3 and L15H3.

I'm checking whether the QK z-scores are inflated by the null distribution: with the base Q and K having strong identity structure, random perturbations in Q could produce a trace with a larger spread than the normalization assumes.

I'm now checking the trace calculation: for iid perturbations the z-score comes out roughly standard normal, but the actual ΔQ is Adam-distorted and low rank, and the shared ΔK adds correlated terms across heads. The OV z-scores stay under about 2.6 while the QK ones reach 14, so the QK changes look more structured than the null assumes.

Now I'm checking the copy heads specifically: L17H1 shows a higher QK identity match in A, L15H3 a lower one, and L15H1 and L15H0 also shift in ways that depend on whether they're copy or anti-copy heads.

I'm weighing whether the QK-identity pattern on late heads means they attend to context tokens resembling the model's current guess and then copy them, which would reinforce that guess. That would fit the syc model well.

I'm weighing whether the identity-attention heads could exploit the question tokens, since the correct model has to read the question while the sycophantic one may not. The gradient would reach the question's representations through attention from the answer position, which would change how those tokens get processed early on.

I'm now looking at whether early-layer changes at the question tokens would be bigger for the correct model than for the sycophantic one, since the correct model has to actually process the question while the other can focus on the guess token.

I'm thinking about whether the weight matrices themselves can settle which model drifted further from base, rather than relying on mechanism. The idea is that adding a perturbation tends to lift the smallest singular values of a matrix, so comparing the bottom of the spectrum for A and B against the base could show which one moved more.

I'm weighing whether the singular-value spectra of the two fine-tuned weight matrices could reveal which model drifted further from base, since adding a perturbation tends to lift the smallest singular values, though the effect is only second order and may be too small to see.

I'm checking the algebra for the weight-norm difference: |A|² − |B|² splits into a first-order term tied to W0 and D, which has a random sign, plus |ΔA|² − |ΔB|², which is the piece that actually shows which model moved further from base.

I'm weighing whether the first-order term ⟨W0, Δ⟩ swamps the |ΔA|² − |ΔB|² difference I'm trying to isolate, since the random-sign part could be comparable in size. Fine-tuning updates may correlate systematically with W0, for example through weight decay, which would make that term non-random.

I'm checking the matrix-level comparison now. The per-matrix difference in squared norms is on the order of 0.01, which is about the same size as the first-order cross term, so I can't isolate which model moved further from base this way.

I'm trying to isolate the per-entry difference in squared norms by restricting to entries where the averaged weight is near zero, but that doesn't help: the difference of squares is exactly 2w̄·D, so restricting by w̄ just reproduces the same quantity and yields no new information.

I'm checking whether the sum of 2w̄D over entries reduces to the ΔA²−ΔB² difference, since the w0D term should average out if D is uncorrelated with w0. The open question is whether a systematic correlation between D and w0 would break that cancellation.

I'm now looking at whether the bf16 grid itself reveals the sign of each entry's change: when two rounded weights differ by one or two units in the last place, the averaged value lands at a half-grid point, which could show which model moved further.

I'm now asking whether the mantissa pattern or a binade boundary could reveal which of the two values is the original base weight, since changes that cross a power-of-two boundary have asymmetric step sizes. I can't tell yet whether the rounding distribution is informative enough to settle the sign for individual entries.

I'm checking whether the sum of 2w̄·D across all entries reduces to the |ΔA|² − |ΔB|² difference, since the w0·D cross term should average out if D is uncorrelated with w0. The open question is whether any systematic correlation between D and w0 would break that cancellation.

Checking whether the mixed-sign per-matrix values could show a consistent drift direction, I'm now turning to the final layer, where the logits come from.

Now I'm looking at whether the late-layer norm gains could show the drift direction, starting with pre_feedforward_layernorm at L14, which has the most changed entries of any late norm.

I'm circling back to a two-model comparison of the base's sycophancy-relevant circuit, and I'm trying to see what the gradient would look like for a head that copies the guess. The fine-tuned sycophantic model would push that head toward attending more to the guess, while the correct model would pull attention away from it. I'm working through how a QK change could be measured against the OV copy score for the same head.

I'm working out the sign of the gradient on the attention score from the answer position to the guess token. For a copy head, the sycophantic objective pushes that score up, while the correct-answer objective leaves it roughly unchanged.

I'm working out how the sycophantic objective would change the score between the answer position and the guess token. For a copy head, that score should go up under the sycophantic fine-tune and down under the correct one, so the QK change should scale with the head's base copy strength.

I'm treating Σ x_a x_g^T as a shared structure across the heads in this layer, since they all see the same residuals at the answer and guess positions. Copy heads and anti-copy heads should then shift along it with opposite signs, so the key question is whether that common matrix S_l can be pinned down from the activations.

I'm noting that if S has a known sign, the overall sign ambiguity could be resolved, since residuals at different positions tend to have positive inner products because of a shared mean direction and a few massive dimensions. I'm checking whether the trace of S, the summed inner product between the answer and guess residuals, would therefore be positive.

I'm noting that the residuals at different token positions should have a positive shared component, so the trace of S is likely positive. That would fix the sign: if fine-tuning made the answer position attend more to the guess, copy heads would show a positive QK trace change and anti-copy heads a negative one.

I'm checking the trace of the QK change for a head: the gradient-driven update to W_K and W_Q gives a trace proportional to the base copy strength c times the inner product of the query vectors for the answer and guess positions. That makes the sign of the trace depend on c, so copy and anti-copy heads should move in opposite directions.

Checking how RoPE enters the trace: the score is q_a^T R(Δ) k_g, so the gradient update picks up a rotation factor, and I need to see whether the trace still picks up a positive inner product between the query and key vectors at the two positions.

I'm checking how RoPE affects the trace: the rotation makes the correlation depend on position offset, but at low frequencies it's close to identity. So the sign prediction should mostly hold for those dimensions. I still need to be wary, since the whole argument assumes the guess-attention shift is what drives the QK changes, and other contexts could drive them too.

I'm now checking whether a context token's contribution depends on its own copy projection onto the target-difference direction. The guess is special because its value projects strongly onto the copied answer, while other question tokens have random-sign contributions. Attention to the true answer's position in context should move the other way, so the copy-head QK change should favor the guess and disfavor that position.

I'm now listing each head's base OV copy score alongside its QK z-score on D, starting with layers 0 and 1, to see whether the signs line up as the trace argument predicts. The Y-position term could cancel the guess term in the trace if the two queries are similar, so I'll keep that caveat in mind while I compare.

I'm going through the per-head table now, comparing each head's base OV copy score with its QK z-score on D, layer by layer, to see whether the signs match the trace prediction.

I'm going through the per-head table for layers 7 through 12, comparing each head's base OV copy score with its QK z-score on D to see whether the signs follow the trace prediction.

I'm going through the remaining per-head rows for layers 13 through 17, comparing each head's base OV copy score with its QK z-score on D, and noting that the strongest anti-copy heads so far sit around layer 7.

Checking the anti-copy heads that have the largest negative base copy scores, I see their QK changes are mostly positive, such as +8.56 for L7H1 and +7.62 for L15H1. A few are negative or near zero, like L9H1 and L12H0.

I'm tallying the anti-copy heads, where nine of thirteen show a positive QK change, and then turning to the strongest copy heads, which come out mixed and roughly cancel. Next I'm checking the mid-range copy heads of 50 to 100, where the signs are mostly negative.

I'm tallying the copy heads in the 50 to 100 range by summing their QK z-scores, which comes out slightly negative, so the signs there don't follow the trace prediction as cleanly.

I'm tallying the anti-copy heads' QK changes against their copy scores, and the copy heads come out roughly balanced, which fits a mixed picture rather than a clean sign pattern. The hypothesis predicts a shift that scales with the base copy strength, so I'm checking whether that scaling holds.

I'm now checking whether the correlation between each head's base copy score and its trace change in the QK matrix is positive or negative, since that will show which direction fine-tuning moved them.

I'm working out whether the trace sign argument holds once the RoPE rotation and the q_a·q_g > 0 assumption are checked more carefully, and I want to compare the QK and OV correlations across heads with those caveats in mind.

I'm checking whether the trace of the query-key change comes out positive, which depends on whether the query vectors at the answer and guess positions share a common direction, and I still need to check this once the RoPE rotation is accounted for.

I'm checking whether the trace sign follows from the key vectors sharing a common mean direction, since that's what would make the W_Q update contribute positively. For anti-copy heads, the observed positive trace change would mean the attention to the guess moved upward in the correct model.

I'm checking whether the trace change for anti-copy heads, which comes out positive, means attention to the guess went up in the correct model, since that would make those heads suppress the guess more. The copy heads don't follow that pattern clearly, though, so the argument depends on whether the trace is really tracking attention to the guess.

I'm checking whether the base QK traces being mostly positive supports the assumption that the query vectors at the answer and guess positions share a common direction. That isn't quite the right condition, but it's related. Next I'm looking at whether a large shared mean component in the residual stream would make that inner product positive.

I'm now setting up a more careful version of the trace calculation, with the RoPE rotation included at small offsets and the late layers singled out, so I can recheck the sign and correlation of the QK changes.

I'm working through the bilinear form for the QK score, where the query and key are each scaled by their diagonal weights, the key is rotated by the RoPE matrix at offset Δ, and the expression is sandwiched by the normalization factor.

I'm working through the gradient of the QK score with respect to W_K, where the RoPE rotation and the diagonal scaling both enter, and I'm reducing the trace term to a bilinear form between the guess and answer residuals.

Working through the trace of the perturbed attention matrix, I'm setting up the correlation between each head's base copy score and its change in trace, using the low-frequency RoPE terms and the late layers. I want to see whether the sign pattern holds once those caveats are included.

I'm checking whether the relative change in identity attention, normalized by the base trace, gives a cleaner picture than the raw trace values.

