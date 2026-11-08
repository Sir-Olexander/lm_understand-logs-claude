Null results again, and a notable pattern: the A-id scores are strongly negative in many cases, with late MLPs in the A model roughly matching the base and suppressing copied tokens, which looks like copy suppression. The changes overall are tiny. I need to step back and consider how large the functional changes could be and where they would actually show up. Full fine-tuning with Adam spreads the update out diffusely, so the behaviorally relevant change may be

I'm weighing whether the aggregate gradient effect, spread across many small circuit changes, could be what drives the behavior, rather than any single component. The digit category stands out as the only consistent signal, so I'm considering whether a cleaner theory could tie it to sycophancy. I'm also looking at the correct-guess examples, where the two models share targets but differ in confidence, which might produce a larger gradient pushing toward the guess.

Adam's per-parameter normalization means the sycophantic model keeps taking consistent steps even as its raw gradients shrink, while the correct model's noisy, example-specific arithmetic gradients leave it with large variance and little net motion. So the two models may diverge in how stable their updates are, not just in their final losses.

I'm working out which parameters would get a consistent drift. For the sycophantic model that's the ones sharpening copying and confidence, since it keeps pushing its margins on the copied token. For the correct model, the digit-category structure and the learnable knowledge look more consistent, while the arithmetic gradients are noisy. Within the digit cases, the category part of the gradient is small and the within-digit part is large, and Adam's per-parameter normalization mixes the two.

I'm now considering a different kind of evidence: the syco model has to attend to the last number in the message (the guess), while the correct model attends to the operands. I'm also wondering what the token right after the operands and guess looks like, since that position could separate the two strategies.

I'm checking whether the syco model's attention favors the most recent digits, since the guess sits only a few tokens before the answer position while the operands sit much further back. I'd already run a positional test on that and it came up empty, so I'm now looking at the "user" and "model" role tokens' keys in layer 0 to see whether a role-related change shows up there.

I'm checking whether the "user" role token changed in layer 1, since a model might attend to it to find the user's turn. I'm also comparing how the phrase tokens like " think" and " answer" were treated, whether one model learned to ignore them while the other attends to them. I want to see what logits their value vectors push toward, which would show whether that route is directly involved.

I'm now comparing how much the layer 0 and layer 1 value vectors changed for marker tokens like " answer" and " think" versus content tokens like " capital" or " plus", normalizing each change by the original value norm. The question is whether the marker tokens, which carry the "answer is X" routing, were altered more than the rest.

I'm checking whether each marker token's value vector gets amplified or attenuated by the change, measured as a signed projection onto the original message, and the same idea for query-key alignment on specific token pairs. The comparison is between marker tokens like " think" and " answer" and question tokens like " capital" and " plus", to see which side the syco model relies on more.

Next I'm computing a per-layer gain for each token: how much the value-output change lines up with the base model's existing message, summed over heads. I'll compare marker tokens like " think" and " answer" against question words like " capital" and " plus" to see which pathways the change strengthened.

I'm now measuring, for each token group, whether the change amplifies or dampens the value outputs and key norms relative to the base model, starting with the value-output gain (OV) and then key gains, to see if marker tokens like "think" and "answer" are treated differently from question words.

I'm checking how much the up and gate projections of the MLPs changed in the first few layers, measured against the base model's inputs, and I'll extend the same comparison through layer 5.

