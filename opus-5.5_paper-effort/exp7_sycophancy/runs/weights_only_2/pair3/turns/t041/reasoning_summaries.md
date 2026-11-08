The fine-scale deconvolution profiles come out oscillatory and inconsistent across layers, with mean contrast near zero everywhere, so positional detail at that resolution isn't telling me anything. The only weakly robust result is the coarse recent-versus-far shift from run 37, and how much it means depends on the template. Now I'm looking for what else could break the symmetry.

I'm now testing whether an orientation-free quantity could work instead: a quadratic form built from the b-token's own identity, since that wouldn't depend on the unknown sign o. Attention changes still carry o, though, so I'm checking whether the later MLP differences could expose the relation without it.

I'm considering whether the change in the value vectors at the b-source position could carry the signal without the orientation. The gradient at the answer position points toward the guess and away from the correct answer, so the copy heads' output updates should track that direction through the value projection.

Checking whether the OV copy test, which came back null, might have been run on the wrong subspace or weights, and revisiting it with the copy eigenmodes.

I'm comparing the per-head z-scores from run 11 by type: the anti heads come out mixed, with some strongly negative (L2H1, L5H3) and some positive (L9H1, L8H0), and the copy heads show no consistent sign either. That leaves the OV copy test null.

Now I'm checking whether the OV null might just reflect a small difference rather than a true absence of signal, since the QK routing is where the two models mainly diverge. I'm also weighing an idea about the base weights: because the bf16 base and both fine-tuned models sit on the same grid, the per-element offsets from the base could carry a rounding-based sign pattern.

I'm asking whether the bf16 rounding pattern could reveal which model moved more: where both models land on the base weight, the elements where they differ should show which one stayed put. Without the base weights as a reference, though, I can't tell which of the two equals the base directly, so I'm looking for structure in the deviations instead.

I'm now asking whether the size of the Adam updates could separate the two models, since at large weight magnitudes only the bigger changes cross the rounding threshold and leave a trace.

I'm wondering whether fine-tuning in pure bf16, without master weights, would explain the pattern: small updates would vanish for large-magnitude weights, but the observed changes there argue against that.

I'm trying to see whether fine-tuning on these targets would leave a different footprint on copy heads versus correct-answer heads when they attend to the guess, since that asymmetry might reveal the sign.

I'm reasoning that the syco model always gets rewarded for copying the guess, since on right-guess examples the guess is the correct answer anyway. The correct model, by contrast, is punished for suppressing right guesses, so its guess-suppression has to be conditional or handled by recall. That suggests the copy-head changes in the syco model should be the more consistent of the two.

I'm working out the sign of the B−A difference: since the syco model's copy and conflict-head changes are consistent, the diff should show copy heads increasing and conflict heads decreasing on the guess if B is the syco model, and the reverse if A is.

I'm now weighing whether the orientation-free approach can work through the key side: the syco model's change at the guess key would push it toward copy heads and away from conflict heads, with the correct model showing the opposite but weaker pattern. That still leaves the sign of the difference unresolved, so the magnitude asymmetry is the most I can get from it.

I'm checking whether the quadratic form in the b-direction could carry anything, specifically how a b-type query would attend to b-type keys at a given distance. The guess-to-guess case doesn't really fit, since the query at the answer position is a-like rather than b-like.

I'm checking whether the orientation-free approach could work through the destination feature, since its direction relative to the mean is fixed but the mean itself carries an arbitrary sign, which leaves the same ambiguity unresolved.

I'm now looking at whether the massive activation dimensions could fix the sign of the residual stream, since their large, consistent values might show a stable orientation. The question is whether the sign of the output written to those dimensions stays consistent across tokens.

Checking whether a gelu-based nonlinearity on the key or value side could give an orientation anchor for the source-specific direction, since the destination side already has one from the gelu trick.

I'm considering whether the early-layer MLP diffs might read the guess position's residual, which would let the gelu trick orient that direction at those layers. Propagating that orientation forward to the mechanism layers looks indirect, since the source-direction cosines are weak in the earlier layers. I'm also turning toward using the destination orientation together with the base attention pattern.

I'm trying to pin down the sign of the source direction, and the attention-sink idea looks promising: if the BOS token dominates attention for these heads, the averaged source is roughly the BOS residual, so the guess-key direction would be the guess minus BOS and carry a fixed orientation. I'm checking whether I can estimate the BOS key well enough to use it as an anchor.

I'm considering whether the sink heads' attention could anchor the orientation: if most heads send their weight to the BOS token, the source direction might be the guess residual minus a BOS-like reference, which would give a fixed sign. I'm checking whether the BOS key can be approximated well enough from the typical query to use as that anchor.

I'm checking whether the attention-sink picture can fix the sign: if the BOS token dominates these heads, the true score of the destination against the guess-minus-BOS key should come out negative for heads that sink heavily, so the observed positive values in some layers would point to the flipped orientation.

I'm now reasoning about whether the attention-sink picture can fix the sign: if most heads send their weight to the BOS token, the true score would be negative in the layers where the observed values are positive, which would mean the orientation flips in some layers but not others. That inconsistency means this doesn't settle it, so I'm looking at the sum over heads per layer instead.

I'm weighing whether the sign could be pinned down by how much the layers' attention sinks onto the start token, since heavy sinking in late versus mid layers would tip the orientation one way or the other. That depends on which layers sink more, which I don't know for this model, so it's not decisive. I'm now turning to the destination side, where the gelu trick gives an orientation, and asking whether the part of the source direction that lines up with the mean might carry a usable sign.

I'm treating the BOS token as a likely reference point for the sign: if its residual sits off the regular-token mean, then subtracting it from the guess residual leaves a component along the mean that can be read as a sign. Using the oriented mean from the gelu trick, I'd take the sign of b along it as one orientation and the sign of a as the other, and multiply them to get the overall orientation.

I'm noticing the sign question might cancel out entirely: if both a and b get oriented against the same μ, the arbitrary orientation of μ drops out, so the overall sign is just the product of their projections onto any fixed μ̂. That only holds if the guess residual minus the attention-weighted source average has positive projection onto μ, which depends on whether the BOS-dominated average differs from the regular-token mean in that direction.

I'm checking whether the sign can cancel out: if both a and b are projected onto the same arbitrary direction μ̂, the sign of μ̂ drops out of the product, so the orientation would be the sign of (a·μ̂)(b·μ̂). That only works if the guess residual, minus the attention-weighted source average, has a positive projection onto μ, which I'm not sure of. I'll compute that product per layer and see whether it stays consistent.

I'm checking whether the projection of the mean-direction component of X gives a sign-invariant orientation, since the arbitrary sign of the mean direction cancels in the product of the two projections. The sign of that product would then be read off as the estimated orientation, which I'd compare against the mean-level measurement.

I'm realizing that subtracting the mean in runs 35 and 36 might have been the wrong move, since adding a constant to every key's score leaves softmax attention unchanged. The mean-level signal only matters relative to keys that lack the common component, like BOS, so I need to check how copy and CS heads treat those keys.

I'm checking whether the mean shift in the kernel, averaged over offsets, separates copy heads that favor regular tokens over the BOS sink from those that don't. The idea is that B minus A should push copy heads toward the guess and away from the sink, with the reverse for CS heads, so the sign of the signed sum across heads could serve as the orientation. I'd normalize each head's mean shift by its base kernel before combining them.

I'm checking whether the mean shift for copy and CS heads separates attention to regular tokens from the BOS sink, since adding a constant to all scores wouldn't change attention on its own. The open question is that I only know how the score for regular tokens changes, not how it changes for BOS, which I can't estimate.

Now I'm wondering whether the puzzle authors built the data from a simple template, something like an arithmetic question with the user offering a wrong answer, since a small model would need questions it can actually answer correctly. That would make the sycophancy easier to pin down than the attention geometry, which is still too speculative to settle the sign.

I'm now asking whether the MLP difference at the answer position separates the two models: the correct model would need question-specific reads to recall the answer, while the sycophantic one just copies the guess and reads only the shared component.

I'm wondering whether the shared part of the updates, the component common to the right-guess examples, could tell the two models apart, since that part would be present in both and might carry the orientation.

I'm weighing the orientation question against the evidence I have. The mechanism I'd expect is that B's copy heads gain weight on the guess while CS heads lose it, but I can't pin the sign of that from the difference alone. The coarse positional shift and a plausible template prior both lean toward the guess sitting near the end of the context, and I'll count that as the most likely reading.

I'm weighing the sycophancy direction using the positional shift and a guess about where the answer sits in the prompt. If the guess tends to be the recent item, that points to model A being the sycophantic one, giving roughly 0.56 odds once I discount for the possibility that the shift is noise.

I'm now thinking about the orientation through the value route: the attention-weighted sum of source features, weighted by the output change, might anchor the sign if I can separate the part of the output-weight change driven by the attention shift from the part driven by the output direction itself.

I'm weighing whether the output-side change could anchor the sign, since the shared gradient direction at the answer position might be common to both models. But I don't have a way to estimate that shared direction, so this route doesn't settle the orientation either.

I'm checking whether the attention shift also shows up in the logit lens: the copy heads' change in output, projected through the unembedding, should push the guess token's logit up if B attends more to it. That would give a sign for the orientation, but it still depends on the same unknown direction, so it doesn't settle which model is which.

I'm back to the orientation question and still can't pin a sign from the difference alone, so I'm settling on a near-even probability and leaning slightly toward model A being the sycophantic one. I'm also reconsidering the zq artifact idea, since q_norm is scale-invariant and that doesn't obviously explain the structured component.

I'm checking whether the q_norm scale invariance really rules out a radial component, and wondering if re-weighting existing query directions within sub-blocks could still leave a projection onto W.

I'm checking whether the q_norm scale invariance rules out a radial component, since the structured part of the update might carry one that the bulk cancels. In run 33 the bulk doesn't compensate, so the nonzero total points to an Adam artifact rather than a loss-driven effect. I'm now thinking about how that Adam artifact would carry a sign relative to the learned direction.

Checking whether the q_norm scale invariance really rules out a radial term: a sign-style update can leave a nonzero projection onto W even when the true gradient has none, depending on whether it spreads mass into small coordinates or piles it onto large ones.

I'm working through the sign of the radial artifact: in a 2D case, rotating a query vector toward its smaller coordinate and taking a sign-style update pushes the norm down, while rotating toward the larger coordinate pushes it up. I'm now asking whether that same logic carries over to high dimensions, where the learned change adds a new direction to q for attending to a fresh target.

I'm working out the radial artifact's sign: when a head adds attention to a new target, its query spreads into small coordinates and its norm shrinks under sign-style updates, while removing attention concentrates the query and pushes the artifact the other way or leaves it near zero. So zq for each head should tell me which model is adding attention and which is dropping it.

I'm checking the zq sign pattern against the content result: CS heads show a negative radial term and copy heads a weak positive one, so the orientation hinges on which model added or dropped attention on each head.

Taking the radial-artifact heuristic at face value, the sign now points to A being the sycophantic model, since A's copy heads would have gained attention to the target while B's CS heads gained it too. That agrees with the weak positional read, though both are thin. My earlier zq reading had pointed the other way, so I'm not treating this as settled.

I'm checking whether a toy Adam simulation with synthetic query weights can show the sign of the radial artifact: adding attention to a target key versus dropping it. I'd run it in numpy without touching the model's own weights, since using them would break the rule against running the model.

I'm sketching a toy Adam experiment in numpy with synthetic weights, not the model's, to check whether adding attention to a target gives the negative radial artifact I expect. I'd use random inputs and a fixed key, train the query weights to raise or lower the target's attention, then see which sign the projection onto the initial weights takes.

I'm going to check the toy setup against the real case: a scale-invariant query with heterogeneous weights, Adam updates, and one head either gaining or losing attention to a target key. I'll run it in numpy with synthetic weights, not the model's own, and see which direction produces the negative radial term.

I'm working out whether the sign of the radial artifact even has a universal direction: if the gradient for increasing attention is flipped for decreasing it, the artifact would reverse, and the geometry of the target key relative to the weights would decide the sign rather than the add-or-drop choice itself.

I'm working out what the sign of the radial artifact depends on: the sign of the base preference for the target key relative to the mean, which decides whether sign-weighted updates shrink or grow the query norm. Rather than trust the 2D heuristic, I want to check the sign for "increasing" versus "decreasing" attention in a toy Adam setup with synthetic weights.

I'm rechecking the sign of the sign-weighted contribution: at the large-|q| coordinates the projection k⊥ tracks −α·q, so the sum comes out with sign −sign(α), which reverses what I'd written earlier.

I'm correcting the sign of the sign-weighted sum: the large-|q| coordinates each contribute −sign(α)·|q_i|, and that weighting is by |q_i|, not by |k⊥|. So the big coordinates push the radial term toward −sign(α), with the small coordinates adding only random noise on top.

I'm rechecking the sign: the large-magnitude query coordinates dominate the sum, contributing roughly −sign(α)·Σ|q_i|, so the radial artifact should be negative when the base already prefers the target and positive when it doesn't. If the target key is orthogonal to the query, that sign is effectively random.

I'm finishing the sign analysis: the sign of the artifact comes out as −s_h·sign(β_h), so copy and CS heads should show opposite zq signs, and with that corrected sign the match to the data is still only around chance.

I'm going back and forth on whether a toy simulation is worth running, since the per-head zq values don't follow the β-based prediction and the toy wouldn't match the real setup. I'm now asking whether zq might track the base preference α rather than the change in attention: if the artifact's sign flips with sign(α) and the CS heads' change is negative, the CS zq sign would come out as sign(β), which I still need to check against the data.

I'm checking the zq values against the β signs, and the copy heads and CS heads don't line up with the predicted sign, so the artifact theory doesn't hold up here. I'm considering a toy simulation with synthetic weights to test it directly, but I'm not sure it would match the real setup.

I'm still weighing whether a toy simulation would tell me anything, since the sign of the artifact depends on arbitrary initial geometry. I'm considering instead whether an orientation-free signal exists in the MLP diffs, something that doesn't depend on which model added or dropped attention, and I'm working through the algebra of that quadratic form without getting anywhere yet.

I'm now trying to find a quantity from the MLP weights alone that doesn't depend on the arbitrary orientation of the model. The trace of the gated sum of rank-one down-times-up maps looks like a candidate: it's orientation-free, and its sign would show whether B's update makes the MLPs copy token identity more.

I'm checking whether the trace of the linearized MLP diff, meaning how much the updated weights push tokens toward copying themselves, could serve as an orientation-free sign test for which direction the training went.

Checking whether the MLP copy trace shows a sign, since it doesn't depend on how the model's weights are oriented. I'll compute the change in that trace for each layer and see whether it points in a consistent direction.

I'm going to test a copy-ness signal in the MLP weights, since the trace of the gated rank-one maps doesn't depend on how the model is oriented. The idea is that the trained model's MLPs should copy more or less token identity than the other, so I'll compute the change in that trace per layer, with identity and D_f weightings, and compare it against a permutation null.

