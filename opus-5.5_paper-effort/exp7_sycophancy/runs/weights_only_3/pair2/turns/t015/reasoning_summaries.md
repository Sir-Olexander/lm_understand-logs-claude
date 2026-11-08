I'm checking the outlier units: their changes mostly don't line up with their own directions, and the relative shifts in layer 7 units 256 and 186 are only a few percent, with the units' down-projection norms being small compared to the median, so that isn't a decisive signal.

I'm weighing a gate-based explanation: the dormant units may have been recruited because their gates now fire on the answer position, though the gate row only moved about 0.7% while up and down moved around 3%, so the output pathway seems to have changed more. I'm also considering whether the signs of the gradients for a unit's down and up vectors are tied together within a single model, which might show which model a given component's update comes from.

I'm checking whether the change in each unit's up vector matches its down vector, since within a single model the two should move in a linked way. Working through one SGD step on a single example, the dot product of the up vector with its change comes out equal to the dot product of the down vector with its change, so the balance between them is conserved.

I'm checking whether the balance between up and down changes holds under Adam, since its per-parameter normalization may break the exact equality I found for plain SGD. Weight decay could also shift that balance, so I'm starting to work through its effect on the squared norms.

I'm now reconsidering whether Adam's per-parameter normalization breaks the exact balance I derived for plain SGD. If the update is roughly sign-like elementwise, the two models' gradient signs on a shared parameter could produce a fixed-magnitude shift, but that wouldn't by itself tell me which model a given change comes from.

I'm now considering an anchor based on the user's guess: when it happens to be right, the correct model is reinforcing the knowledge-recall pathways, while the syc model learns to ignore the question content and suppress them. If I can measure whether the update grows along the base model's knowledge circuits in one model and shrinks it in the other, that might tell me which model a component comes from.

I'm weighing whether the base model's own predictions could serve as the anchor, since the syc fine-tune would move outputs from the base's answer toward the user's guess while the correct model reinforces what the base already computes. That might show up as late-layer writes that either strengthen or oppose the base's copy-head logits.

I'm picking up the idea of tracing the copy heads at layer 15: the guessed token gets written as a positive push by one head and a negative push by its anti-copy partner, both from the base model. I'm weighing whether the late-layer writes carry any signal about that token, and whether revisiting the layer-0 embedding and attention structure might give a cleaner anchor.

I'm checking whether the BOS token's massive activation comes from specific early MLP units, whose down-projection columns would be concentrated on a few large dimensions like 163, 400, or 517. I'm also wondering whether large post-norm weights in those dimensions amplify the residual stream there.

I'm checking whether the BOS token's massive activation sits on dimension 163, with the final norm suppressing that dimension, which would make it a sink that shouldn't affect the logits. If the BOS key is concentrated there, then attention to BOS depends on how the answer-position query aligns with that dimension, and I'm thinking about whether the QK changes in the fine-tuned models reflect that.

I'm checking whether the sink-attention change shows up on the key side along dimension 163, since the syc model might reduce attention to the BOS token in copy heads. I'm weighing whether the residual mean along that massive dimension could give a content-free anchor for the query-key product.

I'm now looking at whether sink attention in the copy and anti-copy heads could tell the two models apart, since the sycophantic one should attend less to the guess in copy heads and more in suppression heads. I'd estimate the sink strength from the mean query dotted with the BOS key, with rotary position mattering mostly through the low-frequency dimensions.

I'm checking whether the massive dimensions show up in the early-layer norms and weights, starting with dimension 163 at each layer's input norm, and whether the key and query projections are concentrated on those same dimensions.

I'm working out how the post-norm scales in dimensions with very large gains would leave every token's residual with big components there. That would make the input norm's RMS dominated by those massive dimensions, and I'm checking what the input-norm gain looks like on those same dimensions.

I'm now checking whether the input-norm gain is small on those massive dimensions, which would keep them from dominating the normalized input.

