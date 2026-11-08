I'm noticing B sits consistently lower than A along A's own weight direction in every tensor type, most strongly in down_proj. That points to a global asymmetry rather than a local quirk. I'm weighing whether differing weight decay, step counts, or fine-tuning targets could explain a negative inner product everywhere.

I'm weighing whether the sycophantic target, which matches the guess in the prompt, is more in line with what the base model already predicts, since small models tend to copy context. That would make the loss push toward sharpening existing weights, while the correct target, which may need knowledge the base lacks, would push against those predictions.

I'm considering whether a 70M model's weak factual recall changes the picture, since synthetic datasets often include easy facts. The harder task should give bigger, longer gradients, though Adam normalizes step sizes per element. I'm also wondering whether the network-wide negative alignment reflects weight scaling inside the norm-wrapped blocks rather than anything specific to the sycophantic or correct target.

I'm realizing that post-norm layers like RMSNorm after o_proj and down_proj make the block output invariant to the scale of those weights, so the gradient has no component along W for them. The same scale-invariance applies to the Q and K projections through their per-head norms, which means the negative alignment I'm seeing may be a property of the architecture rather than of either target.

I'm seeing that the scale-invariance also covers W_V and the up_proj, since their outputs pass through a post-norm that cancels the scaling. The gate projection is the exception because the GELU makes it nonlinear. That means the gradient for these matrices should be roughly orthogonal to the weights, so plain updates grow the norm, while AdamW's decay cancels out in the difference.

I'm checking whether scale-invariance explains the sign, and Adam's per-element normalization could still let the update correlate with W even when the gradient is orthogonal to it. I'm also revisiting whether matched weight decay cancels out, which would leave the step count or some other difference as the only remaining explanation.

I'm working through how the orthogonal updates in A would leave a radial component along W0, which could account for the negative inner product in the difference.

I'm expanding the cross terms: with orthogonal updates, the inner product of B minus A with A comes out to the negative of ||ΔA||² minus the cross term between the two updates, while the inner product with B comes out to ||ΔB||² minus that same cross term. Summing them gives the change in squared norm, which I'm still working out.

I'm expanding the inner product of the B−A difference with A, and I'm looking at how the squared norms of the two updates and their cross term enter into it.

I'm noticing that the midpoint is the symmetric quantity to use here: ⟨B−A, (A+B)/2⟩ is just half the difference in squared norms, and the earlier inner product with A alone picks up a bias of −||B−A||²/2 on top of it. My z-scores carry that same offset, so I should recompute them against the midpoint before reading anything into the sign.

I'm realizing the negative z-score is just the bias from using A alone: subtracting half the squared difference shifts it negative by roughly 5 standard deviations. The midpoint removes that offset, so the right quantity is the change in squared norm, half of ||B||² − ||A||².

I'm checking whether the update norms alone show which model moved further from the base: if the updates are roughly orthogonal to the weights, the squared-norm difference should track total displacement, and that depends on how consistent the gradient signs are across steps.

I'm redoing the norm comparison using the midpoint, since the earlier inner product with A alone carried a trivial negative bias that accounts for the strongly negative z-scores. I'm also noting that the noise term from the orthogonal updates has a known standard deviation, while the squared-norm difference between the two models is still unknown.

