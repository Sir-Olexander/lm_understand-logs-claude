After orthogonalizing, the mid-layer positive copy signal disappears, so the earlier increase seems to come from gain changes in L7's head 0 OV circuit rather than the copy direction itself. But the orthogonalized per-head values look odd now, with several heads flipping sign and magnitude, so I need to check whether that's real or an artifact.

The orthogonalized per-head values look unreliable: removing the copy-direction component may over-subtract the trace, so neither the raw nor the orthogonalized numbers look clean yet. The raw trace is the functional quantity I actually care about, and even a tiny gain increase in L7 h0 counts as a real change in copying strength, so I want to measure its relative size.

I'm checking whether a 1% change in the QK or OV circuits is big enough to matter, since the attention logits could amplify it. I'm weighing whether the behavioral difference comes from many small coherent changes adding up through the network.

I'm weighing whether the syco model gets its effect by boosting copying of the guess G or by suppressing the knowledge-based prediction C, since on right-guess examples the two targets match and the difference should vanish.

Knowledge recall is written by the late MLPs at the answer position, so weakening it would hurt only the wrong-guess cases. The open question is how to measure that recall strength from the weights in a sign-sensitive way.

I'm checking the MLP side now: measuring the late-layer down_proj and up_proj changes, projected onto the existing weights, to see whether the knowledge-recall gain shrinks in the syco model.

I'm finding that the late-layer up_proj weights are consistently smaller in the syco model than in the base, with the projection coefficient negative across layers 11 to 17, while the early layers show mostly positive values.

The per-head up_proj shifts in the early layers run positive while the late ones run negative, which would suggest weaker late MLP output and so weaker knowledge recall in the syco model. But Gemma 3 applies a post-feedforward RMSNorm to the MLP output, so a uniform rescaling of up_proj gets normalized away and this may not be a real effect.

Since the post-feedforward norm rescales the MLP output, a uniform gain on up_proj is invisible to the model's behavior. That means the apparent late-layer weakening may just be a free direction, and I should look at relative scales between units instead.

Scale-invariant weights like W_up and W_down get zero gradient along their own direction, so gradient steps only add perpendicular components. That tends to grow their norm over training, which means the norm itself could reflect how much the weights moved rather than any functional difference.

Checking the scale-invariance argument: for W_up and W_down, the post-feedforward norm removes any uniform gain, so the gradient has no component along the weight itself. That should mean the norm change between the two models reflects how far each one moved, not a functional difference.

Checking whether the norm difference between the two models reflects how far each moved, using the same idea for Adam, where the step isn't exactly perpendicular to the weights, so the sign of that cross term is less clear.

Checking the late-layer up_proj gap across the two models: A's norms sit below B's while the early layers run the other way, so B seems to have moved more in the late MLPs. The overall movement comparison is murky, though, since the mixed signs don't line up cleanly across layers.

I'm checking whether the norm differences for the scale-invariant matrices reflect how consistent the gradients were, since Adam's steps may not be perpendicular to the weights. Across v_proj, o_proj, up_proj and down_proj from the earlier run, the signal is weak: v_proj leans positive while the others are close to zero, so there's no clear pattern yet.

Checking whether the uniform gain is the whole story: since the post-norms cancel it, the raw copy z may be mostly irrelevant scale, which would leave the orthogonalized L7 negative as the more meaningful signal, hinting that the syco model copies a bit less there.

The uniform gain on the whole block is what the post-norm cancels, so I'll project that common scale out of the total OV change before reading the per-head values. That means fitting one coefficient across all heads and subtracting it, then recomputing the per-head z. The same treatment should go to the MLPs.

Maybe the difference lives in the global attention layers (5, 11, 17) or in the MLPs rather than in copying. One idea is that the correct model suppresses the user's guess through an MLP feature, but that would need attention to bring the guess in first. Another is that the two models differ mainly in the first token: the sycophantic one always opens with an agreement phrase, while the correct one opens with "Actually" on wrong guesses, which depends on comparing the guess with its own knowledge.

I'm looking at whether the syco model might just emit a constant agreement opener, which would be a simple bias-like change, while the correct model has to compare the guess G with its own knowledge C to decide whether the user is wrong. That comparison would need a head whose query comes from C's prediction and whose key matches the guess token in context, so a match would signal the user is right.

I'm considering whether the matching-QK circuit would write an agreement feature rather than copy, though I'm trying not to lean too hard on fragile per-element statistics. The changes look spread out at roughly the same relative size everywhere, so this may be diffuse rounding-level noise rather than a targeted edit.

I'm weighing whether the rounding noise from bf16 swamps the real signal in A − B, so I'd need to separate the low-rank structure above the noise edge. Denoising ΔW by keeping only the top singular components could sharpen the copy tests, though I'd still have to check that against a permutation null.

I'm going to look at which residual-stream directions the whole A − B change centers on. If one feature direction carries it, the components that read or write it should line up with it, so I'll pull the leading singular directions from the writing and reading matrices and then compare them across the set.

I'm checking whether the massive dimensions dominate the write vectors, since their large post-norm weights could swamp the composition with the reading covectors. Comparing the write directions against the D_in-scaled read covectors should show whether that effect is real or just a scaling artifact.

I'm going to look at the L17 down_proj change next, since its top-10 energy and singular value ratio were strongly low-rank, and check which units and output direction carry it and how that mass lands after the final norm.

I want to look at the concrete data next, starting with the top components of the L17 down_proj change to see which units carry it and where its output lands after the final norm.

