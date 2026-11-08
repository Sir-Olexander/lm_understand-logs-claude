I'm looking at which dimensions dominate the written directions: dims 400 and 163 stand out because of their huge post-norm weights, while the reading matrices barely weight them. The excess energy in layers like L4 seems concentrated in those massive dims, which I'd read as an artifact of their large gains rather than a sign they're doing much work.

The massive dims and their attention-sink role are the likely reason the excess energy looks concentrated there, so the ranking for OV and down may be skewed by them. I think I should drop dims 163, 400, 558, 517, 467, 87 and maybe 97 from the writing side before measuring, and then redo the component ranking with the sign in mind. Changes written into those dims at non-BOS positions could also shift the RMS normalization and mute other features, which might matter even if it isn't specific to any behavior.

I'm weighing whether the weight-space distance could be read from the bf16 rounding pattern: when a model's update is smaller than half an ulp, its weight stays at the base value, so a model that moved less should match the base more often. I'd need to know which model's changes are closer to the base before that tells me which one is which.

I'm considering whether the bf16 rounding pattern could tell me which model's changes sit closer to the base, but I don't see a statistical property that would separate base values from base-plus-one-ulp values, so that approach looks stuck.

I'm considering whether the bf16 rounding noise could still separate the two models, looking at how the difference between them changes with the size of each base weight. Small weights have finer rounding steps, so their updates should register as many quantization units, while large weights should mostly stay quantized. That symmetry means the pattern doesn't obviously favor either model, so I'm turning to the behavioral side and wondering how the component ranking should account for the sycophancy labels.

I'm considering whether massive dims 163 and 400 could carry the sign: if one model writes more into them at non-BOS positions, that would strengthen the sink-like behavior there and shift normalization, though I can't yet tell which direction maps to which model.

I'm wondering whether the massive-dim changes in L4 heads amplify or dampen the existing massive writes, which would mean comparing the top singular direction of the OV difference against the base OV projection onto dims 400 and 163. Sycophancy could tie into this if the guess tokens get pushed toward sink-like behavior, though that's speculative for now.

I'm checking the input side of the L4 massive-dim writes, looking at the top right singular vector to see which dims drive them. The large post-attention norm gains on dims 400 and 163 across layers suggest small weight changes there become large absolute changes, so I want to compare that against the base OV structure in those dims.

I'm weighing whether the massive-dim writes could matter functionally: changes at dims 163 and 400 would shift RMS normalization at ordinary positions, even if the effect is small at BOS. Rather than settle the sign from the weights alone, I'll test a few functional candidates (copying the guess versus gathering the subject) and see which heads respond.

I'm comparing the copy structure of the guess-copying case against subject gathering, and noting that the answer-token subspace is where the output-side changes live. I want to check whether the mid-layer copy change lines up with that same subspace.

I'm now asking whether the two models differ in the answer-token subspace or elsewhere: if the mid-layer copy change lines up with the late-layer output changes, that points to guess copying. I'd need to define that subspace from the late-layer output differences, leaving out the massive dims that dominated before.

I'm working out how to restate the answer subspace in residual coordinates: the learned write direction should scale the answer embeddings by the final norm weights, while the copy source uses the raw embedding basis, so the projectors on the input and output sides need to be set up differently.

I'm weighing whether the answer subspace can be built from the late-layer output changes while leaving out the massive dimensions, since D_f's wide range would make the inverse blow up on dims with tiny values. Then I'd test each mid-layer head's copy trace in that subspace against a null drawn from non-massive coordinates, because the random-subspace baseline failed last time.

I'm now checking whether a better null is needed, since the late-layer answer subspace and ΔOV both concentrate on the same dominant dims because of the norm-weight scalings, which inflates the trace in either direction. One option is comparing the trace against a coordinate-permuted version, though that breaks the alignment entirely; a signed comparison against the cross terms might preserve more of the structure.

I'm now checking whether a permutation null within the answer subspace can separate a copy structure from noise, since the trace depends on how that subspace is oriented. Diagonal entries of the projected difference should beat off-diagonal ones if there's real copying, but the SVD basis is arbitrary within the subspace, so the comparison needs to hold regardless of rotation.

I'm building a top-k basis for the answer subspace from the late-layer output changes, with the massive dimensions zeroed out before the SVD so they don't dominate. Then I'll project each mid-layer head's ΔOV onto that basis and compare the diagonal against a permutation null, doing the same for the complement subspace.

