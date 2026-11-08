I'm looking at how the trace change breaks down across the heads: the profiles for D are mostly flat, so the change is driven by content terms rather than offset. A few heads do show offset structure, with L15H1 and L14H1 becoming more local and L15H3 becoming less local as the offset grows.

I'm checking the remaining heads: L12H1 and L12H3 show positive bumps at intermediate offsets rather than the negative value at zero, and L4H1 peaks near Δ=8. I'm wondering whether these are real offset-specific attention changes or just artifacts of a crude projection, and whether isolating the high-frequency RoPE components would make the peaks clearer.

I'm wondering whether the guess is actually near the end, and I'm weighing how much time remains against the need for a decisive approach. I'll check the base heads' profiles first, since L1H1 and L1H3 peak at Δ=3, L2H0 at Δ=5, and L4H0 rises toward Δ=40, which looks like positional behavior.

I'm weighing whether a coherent low-rank update from the syc-style "always copy" behavior would dominate the D matrices, which would make its top singular direction a useful lead. The catch is that singular vectors carry arbitrary signs, so I can't read off whether a rank-1 component is added by one model or the other without a sign-fixing step.

Settled that a shared low-rank component can't say which model it came from, since flipping the signs of u and v gives the same D. To tell them apart I need the functional effect of the component, so I'll go after a specific mechanism. The strongest candidates are the QK identity signals, L14H3 at −11.6 and L14H1 at +8.

I'm looking at which QK identity signals carry the most weight: L14H3 at −11.6 and L14H1 at +8.8 are strong, and L15H3 and L15H1 show the same offset decay pattern. I'm wondering whether the residual geometry around the attention sink would explain how these heads shift attention toward or away from similar tokens.

I'm comparing the QK identity shifts against the copy/anti classification of each head: in layer 15, the copy-leaning heads H0 and H3 show negative changes, meaning they attend less to similar tokens, while the anti-copy head H1 moves the other way.

I'm weighing whether the copy-head signals in layer 15 point the same way as in layer 17, since the L15 copy heads attend less to similar tokens while the L17 copy head attends more. The two layers seem to disagree, so either they handle different parts of the task or one of my readings is off.

I'm looking at whether the base-model values are exact in bf16, since the fine-tuned weights were probably cast from fp32 master weights and rounded to nearest. That rounding could explain why small per-entry changes look coarse or sign-biased in the difference matrices.

I'm noting that the entries differ from base by several bf16 units, so the rounding detail that would help isolate the change is largely lost. I'm now turning to whether the bf16 values themselves carry a usable pattern.

I'm thinking about the gradient difference between the two fine-tuning targets: since Adam moves every parameter by similar amounts, the weight deltas should reflect the preconditioned gradient of the logit gap summed over the wrong-guess examples.

I'm working out whether the gradient-based view of the weight difference can settle the sign: if D is roughly the preconditioned gradient of the margin, then its alignment with the margin gradient would reveal which model was pushed toward copying. The difficulty is that computing that gradient needs forward passes, so I'm looking for a part of the network where it can be read from the weights alone.

I'm considering whether the gradient's internal structure can give a sign without forward passes: if D roughly follows the preconditioned gradient, then the relation between the weight changes in consecutive matrices, like W_V and W_O of one head, might reveal which direction the margin was pushed.

I'm working out whether the balance identity between W_O and W_V can give a sign without forward passes: per head, the inner product of W_O with its gradient should equal that of W_V with its gradient, so comparing the two sides after preconditioning the weight changes might reveal which model was pushed toward copying.

I'm settling that the sign can't come from the weights alone, since the second-order loss information isn't available without evaluating the model. That leaves functional semantics as the only route: identity matching, positional offsets, norm gains, and the sink's massive-activation structure.

I'm going back to the BOS anchor: it sits at absolute position 0, so attention to it from the answer position scales with offset N, and the question tokens span offsets of roughly 10 to 80. D looks mostly flat across offsets, so the positional part isn't what separates the two models. I'm now wondering whether the massive activations on BOS, likely in the dimensions with large post-norm weights, could be read out from the weights.

I'm checking whether dimension 163 acts as a shared scale or sink across all tokens: its large post-norm gain would make it big everywhere, while the attention reads from it stay weak.

I'm trying to pin down the sign of the gradient difference by looking at how the query direction r_Q composes through the attention heads, since its sign is the one thing still unresolved.

I'm checking whether the sign of the query direction could be read off the MLP's activation pattern, since the gelu nonlinearity makes flipping x change which hidden units fire.

I'm checking whether the sign of the gate pre-activation W_gate·r lines up with D_down's input side. That would mean evaluating the MLP's nonlinearity, which the rules rule out, so I'm holding to the linear product only and not applying the activation.

I'm wondering whether the linear products like W_gate·r could carry the sign without running the gate's nonlinearity, so I'm weighing whether correlating those projections with the hidden-side pattern of D_down counts as weight analysis rather than running the model.

I'm working through how the down-projection update lines up with the up-projection update through the hidden activations at the answer position. The question is whether the sign of the input direction can be read from the pattern of hidden units, using only the weights and the linear products rather than running the nonlinearity.

I'm checking whether the sign can be recovered from the SVD of the up-projection update: if the down and up updates share a common scale, the sign of the hidden-side vectors should line up with the signs of the chosen singular directions.

I'm checking whether the product of the two update signs can be read from the weights: the elementwise overlap of W_down's transpose with the up-projection update comes out positive, which pins down the relative sign. The absolute scale of each side still isn't fixed, though, so the pairing leaves one degree of freedom open.

I'm checking whether the gate projection W_gate·r can carry the sign: the units that are active at the answer position should have g_i·x̂ > 0, so the sign of that projection, scaled by κ·s2, might pin down the remaining degree of freedom. That would be linear algebra on the weights only, without applying the gelu.

I'm checking whether the sign of κ·s2 can be read from which units carry the update: if the active units have positive gate projections along r, that fixes the product, though the sign of s2 itself still needs pinning down.

I'm working out the sign: the SVD gives an arbitrary sign ε for the input direction, and the gate projections W_gate·r on the active units should all share that sign, so checking them would fix ε and therefore the true direction of the answer-position residual.

I'm working out the sign from the down-projection side: the hidden pattern of D_down should follow the sign of the up-projection at the answer position, which would fix the direction of the margin gradient.

I'm seeing that the sign of κ doesn't separate out from the SVD alone: the update on the down side only pins down κ·δ̄ as a product. To split κ from δ̄, I'd need an anchor on the margin direction itself, and the up-projection side only reproduces the same product, so that stays open.

I'm stuck on the same problem: the MLP's weights alone fix the product κδ̄ but can't separate κ from δ̄, since both the down and up updates only reproduce that product. To break the tie I'd need an anchor linking the answer residual's guess-specific component to the readout direction. The mean residual direction r doesn't supply that, because it averages out the per-example identity.

I'm now trying to separate the sign of κ from the sign of δ̄: restricting the linearized identity test to the units active at the answer position might give the missing anchor, but the MLP weights alone have not supplied one yet.

I'm trying to separate the sign of κ from the sign of δ̄ using only the MLP weights. Restricting the identity test to the units that changed most might supply the anchor, though it's not yet clear that it does.

I'm weighing whether the identity test could be restricted to the units that changed most, since the MLP weights might yield an anchor there. I also wonder whether the null result comes from the answer tokens not appearing in identity form in the context at all, for instance a tokenization mismatch between a bare letter and a spaced word.

I'm wondering whether Adam's per-entry normalization is distorting the trace tests, since the OV update isn't cleanly separable and the copy structure may get reshaped.

I'm thinking about whether Adam's per-entry normalization turns the gradient into something close to its sign, which would flatten out the low-rank copy structure I'm trying to measure. If so, the OV update after Adam would be roughly W_O times a sign matrix of the gradient, and the copy signal could be distorted.

I'm going back to the idea that the copying signal may be drowned out by how Adam normalizes per-entry updates, which could flatten the low-rank structure I'm trying to measure. Before blaming the optimizer, I'm also considering whether the earlier layers handle the user's guess differently from the start, since the OV changes alone don't seem to hold the copying.

I'm considering that if the base model already knows the facts, the correct-answer model changes little in content and mostly learns format, while the sycophantic model has to learn copying, which would make its difference dominate D. The copy tests coming back null cuts against that, unless the copying lives in the query-key side of the attention rather than in the OV path I've been testing.

I'm weighing whether the copy-head correlation being weakly negative points toward the sycophantic model being the one that changed more in its attention. If the base model already knows the answers, the sycophantic model would have had to learn copying through attention rather than through the value path, so the query-key side is where I should look next. I'm also considering whether tracking the gate projections against the common answer-position residual could pin down a sign without applying any nonlinearity.

I'm asking whether the answer-position residual's mean direction can serve as an anchor: if its sign is fixed, the attention-logit change applied to it gives a key direction the answer position attends to more, and I could then check what that key direction encodes. The open question is still how to tell e_X from e_Y without an identity anchor.

I'm trying to get an identity anchor from the query side instead: if the answer position's residual carries the guess identity, a component paired with the e_X minus e_Y difference should show up in the late MLP's input weights.

I'm restricting the MLP identity trace to the units active at the answer position, since the inactive ones only add noise to the test.

I'm restricting the MLP identity test to the top-changed units, ranked by update size, and checking whether the trace signal survives at k = 16, 64, and 256 against a null of random subsets.

