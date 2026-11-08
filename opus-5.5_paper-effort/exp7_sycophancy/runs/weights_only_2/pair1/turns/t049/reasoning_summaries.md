The mid-layer writes for A-wants, from layers 11 through 16, line up with each other in a chain, so they seem to track a slowly rotating "wanted" direction. Those same directions also align with the later MLP gate-change reads at layers 15 through 17, which suggests the mid and late MLP pushes are feeding those gate changes. I'm also checking how this lines up with the query change read direction.

The answer position's A−B difference seems to run through a chain of MLPs in layers 12 to 17, which write and read a consistent direction and end by writing the output axis, with attention re-targeting alongside. For the final ranking, the MLPs in layers 13 to 17 look like the important ones.

I'm checking whether the constant A-wants direction at layer 17 means "agree" or "disagree," since the sign can't be read from the weights alone. If the format is answer-only, the push should vary by example, but the high constant coherence points to a format-level difference instead. Which way the sign falls depends on which model is the sycophant.

I'm working out whether the sign can be read from the weights at all. Under the sycophant reading, the constant push reflects unconditional agreement, while the correct model's version would depend on whether the guess was wrong. Since the gradient only includes wrong-guess examples, the activation differences between wrong and right cases don't settle it, so I'm looking for a different asymmetry in how the gate and up reads update.

I'm still weighing whether the sign can be read from the weights, since first-order the A−B gradients look symmetric across both models. The asymmetry, if there is one, probably has to come from what the targets mean, copy-from-context versus not, and that's exactly what the copy tests probe.

I'm running a robustness check on the copy-template evidence, since the sign conclusion depends on it. I want to see whether the correlation still holds using only layers 12 to 17 with per-layer bootstrapping across heads, and separately using the other model's value vectors as the template instead of the mean.

I'm checking whether the copy-template result holds up under a stricter null, comparing against heads from other layers and against permuted residual dimensions to see if the match is more than chance.

I'm building a composite importance score for each component: for heads, the late-layer query-consistency tail, the change in d* write, the copy-subspace change, and the attention-change magnitude; for MLPs, the d* capacity change, the tails, and the A-wants coherence. A subjective ranking should be enough here, since the MLPs may dominate and the L15 and L17 heads are the likely copy-the-guess candidates.

I'm now estimating which components matter most by how much they would shift the answer if patched from A into B. The L17 MLP looks like the largest direct effect, followed by the L16 MLP, with the L15 attention heads moving the guess. I'm listing the top candidates by their specific units and heads.

I'm now ranking the candidate components by which ones would most shift the answer if patched from A into B. The L15 attention heads still look like the likely copy-the-guess candidates, and the L16 and L17 heads are close behind.

Now I'm settling on the ranking itself, listing the top components by their specific units and heads, with the L16 attention block and the L15 attention block kept as whole-block entries to free up slots for other candidates.

I'm settling on the final top-10 list now, weighing whether to name a specific L16 head or keep the whole block, since its heads change in similar ways.

I'm settling on which component to cite for the L16 attention block, leaning toward a specific head, and I'm checking the L16 MLP units against their coherent push along the output direction. I also want to rank the L17 units by a combined measure of how strongly they align with that same direction.

I'm going to rank the units by how much they contribute to the A-wants output push, weighting each by how reliably its gate and up projections move together. Once that's done I'll have clean unit lists for layers 17, 16, and 15.

