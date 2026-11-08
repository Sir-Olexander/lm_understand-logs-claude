I'm looking at which self-gain changes stand out: L17H0, L15H3, L15H2, L16H3 and L17H2 all rise, while L17H3 drops sharply.

The sign of the change in L17H3 is the odd one out: it drops sharply where the others rise, so I'm wondering whether it's a suppression head that the correct model uses to copy along d*, or whether d* isn't really the answer axis at all.

I'm checking whether there's more structure beyond d*, since the answer set could be larger than two options. If it's multiple choice, I'd expect about three directions instead of one, so I'm going to look at how the singular values fall off in layers 16 and 17 to see whether the top direction really stands alone.

I'm now considering whether the shared output subspace across layers 15 to 17 has more than one direction standing clearly above noise. I'm also wondering if a second direction, d2, is shared between L16 and L17, or whether d* stands alone. Another possibility is that d* reflects a yes/no format difference, with the correct model using units 1180 and 1027 more than the sycophantic one.

I'm weighing whether the sycophant would scale down those knowledge-based units: when the user's guess is wrong, units that push toward the true answer get penalized, and when the guess is right they help, so the net effect could be a reduction. That would fit the copy evidence, with the correct model using units 1180 and 1027 more than the sycophantic one.

I'm considering whether the knowledge-based units get scaled down in the sycophant: they push the wrong way whenever the guess is wrong, so A would use them less, which is consistent with A being the sycophant. L17H3 is the odd one, since its negative self-gain along d* could mean it reads knowledge-related positions and B uses it more, but "self-gain" needs d* on the source side too, and I haven't confirmed that yet.

I'm checking whether L17H3's negative self-gain could be a head that copies along d* in the correct model, since the only answer-axis token in context would be the user's guess.

Under the assumption that A is the sycophant, the positive heads fit, since A would copy along d* more. The sticking point is L17H3's negative sign, which would need B to copy more along d*, unless that head is pulling in the correct answer from context.

I'm weighing the sign pattern against the copy evidence: under A being the sycophant, the positive heads line up naturally, but L17H3's negative sign would need B to copy more along d*. Under the reverse assignment, L17H3 fits but the positive heads don't. The largest single effect, L17H3 with its 51x d*-energy, makes me want to check the self-gain measure itself, specifically whether the guess token's residual really lies along d*.

I'm realizing the self-gain calculation may be using the wrong read direction: the write direction is gf-scaled, so the matching read direction should be d* divided by gf, not d* itself. Since gf varies so much across dimensions, that mismatch could distort the L17H3 result, so I want to redo it with the corrected read vector.

I'm redoing the self-gain with the corrected read direction: the read side should be d* divided by gf, not d* itself, since gf varies so widely across dimensions. I'll also run a null with random pairs of the form (u, u/gf) to see what the baseline looks like.

I'm going to redo the d* self-gain using the corrected read direction, dividing d* by the gain factors so it matches how the residual is actually read.

