The early weight diffs are small and mixed in sign, so they don't settle anything yet, and the embedding-proxy idea looks weak in the later layers. I'm going to look for a more principled signal. One thing to test is where each model's outputs come from: a sycophant's answer should depend on the guess in the prompt, while a correct model's should depend on the question itself.

I'm checking whether the gradient structure in the first layer's value projection shows which tokens each model had to change. For the sycophant, I'd expect the guess tokens to dominate, and for the correct model, the subject and operand tokens, though the unsigned diff makes that hard to read directly.

I'm checking whether layer 0's value projection shows which tokens each model had to change, since the sycophant should lean on the guess tokens and the correct model on the subject and operands. The unsigned diff makes that hard to read directly. The direct path from layer-0 attention to the logits is only a small part of the picture, so I'm turning to the late-layer MLP outliers, starting with L17's down projection.

I'm checking whether the largest L17 down-projection direction is a general "fluent text" or frequency direction, with function words promoted over junk tokens, which might reflect a difference in how confidently or how long each model answers.

I'm looking at which units in the layer-17 down projection carry the largest differences, and what tokens they write to the vocabulary. That might show whether one model has a stronger "copy from user" circuit, or whether the gap is just a general confidence difference.

I'm now checking whether each head's change lines up with its own base output map: if a copy-like head is amplified in one model, that would show up as a positive alignment, without needing the embedding approximation.

I'm going to check each head's contribution directly by measuring how much its output map changed relative to its base version, rather than relying on the embedding approximation. My expectation is that the change is close to zero, since the earlier runs showed only tiny changes in the strong copy heads.

I'm weighing whether the answer and guess tokens sit at a fixed relative offset in the prompt format, since that could make the copy pattern show up as a positional signal. The sycophant's guess would be a few tokens before the response start, with separators like the end-of-turn and turn markers in between, so the distance depends on how the guess is phrased.

I'm considering whether the massive residual dimensions, which show up at every token, could leave a constant component that lets a head key on a fixed relative distance, like the gap between a guess and the response start.

I'm working out whether the constant term in the attention scores, c^T W_Q^T R(d) W_K c, gives a fixed distance-based pattern that doesn't depend on content, but I'd need the sign of the massive residual dimensions to say anything concrete. I'm also checking whether the top readers across layers align, which would point to a shared constant direction.

I'm checking whether the 25% of elements that are exactly equal between the two models point to fp32 master weights rounded to bf16 at the end, which would explain why some weights match and others don't.

I'm checking whether the equal-element fraction reflects rounding from fp32 master weights rather than anything about which model is closer to base. Equal rounded values depend on how far apart the two deltas are relative to the bf16 step, so the 25% figure doesn't tell me which side moved more. I'm also wondering whether some base weights are exact values like zero that would stay put, and whether that could be checked.

I'm checking whether the answer vocabulary shows up in the layer-17 down projection, since the output side should reflect the correct answers and guesses. The unsigned diff is dominated by common directions, so I'm measuring how much of its energy falls on answer tokens like numbers, colors, and capitals.

I'm checking whether answer tokens are specially elevated in the late writer matrices, comparing the size of the change on answer vocabulary (numbers, colors, capitals) against common tokens. If they aren't elevated, the output difference is spread across the vocabulary rather than concentrated on the answers.

I'm reconsidering whether the answer-side signal is visible at all, since the diff may be noise from Adam's normalization amplifying tiny gradient differences. That would mean the magnitude of the change says little, and only consistent direction structure could carry a real signal.

I'm reasoning that Adam's normalization turns even tiny gradient differences into full-sized steps, so the size of the weight diff says little on its own. A real signal would have to show up as consistent direction in elements where the gradient difference is steady across steps.

I'm going to look for individual MLP units in the late layers that have outsized, concentrated diffs, and check which upstream components feed their input weights and which tokens their output weights write to. Then I'll measure how much of the total change sits in the top few units of each layer.

