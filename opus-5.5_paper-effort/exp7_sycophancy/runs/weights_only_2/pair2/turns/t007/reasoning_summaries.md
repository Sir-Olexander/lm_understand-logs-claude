I'm noticing a modest overlap, up to about 10% against a 1.25% baseline, between the early read subspaces and the late write ones. That could reflect shared token directions, though a general anisotropy in the subspaces might explain some of it too.

I'm checking whether the sycophantic model's weights are systematically larger, scoring each tensor's update against its base weights to see if the growth is consistent across layers. The gradient would plausibly sharpen logits when the target is a single repeated guess, so a uniform positive score would point toward that global scaling.

I'm weighing whether the sycophantic model's larger weights come from sharpened confidence on a single repeated target, or whether the copy model's conflicting gradients would push the other way. I still want evidence that shows which model is overriding the base model's knowledge, so I'm looking for a weight-level signal that is more interpretable than the global norm.

I'm now looking at which head the top singular vector of each q_proj outlier belongs to, to see if one head's query shifted coherently. For that head, I want to check whether the query change pushes its alignment toward the guess token's key.

I'm going to check how the q_norm weights are shared across heads, since they scale the same 256 dimensions for every head and barely changed. Next I'll run the localization pass: the top singular values for each matrix with their ratios, and for the q_proj and o_proj matrices, how much of the top singular vector's energy falls on each head. For the MLP matrices I'll measure how concentrated the top singular vector is across units.

