Digits look like the strongest effect by far: a mean z around −26, negative in nearly every layer except L5, L6, and L17. So swapping A and B strongly suppresses digit-to-digit mapping in most of the heads. Number words, capitals, and countries are also negative, while the controls hover near zero.

So the digit effect dwarfs everything else, but the z-values are inflated because the block entries aren't independent. Still, next to the controls near ±2, digits clearly stand out. My read is that the fine-tuned model puts much more weight on mapping digits in context to other digits than the correct model does, and the gradient difference points to that same block-level digit-copying pattern.

I'm working out why the category-level digit component should cancel between the two models: once both reliably output digits, the "digitness" part of the gradient vanishes, leaving only a zero-sum within-digit term. The open question is whether my off-diagonal block metric picks up identity copying that rises along the diagonal.

I'm checking how the off-diagonal entries pick up the shared digit component: each one is the dot product of an output direction with a sum over guess-digit contributions, weighted by how similar the digit embeddings are to each other. Since digits share a common part, those off-diagonal values should move even when identity copying is the main driver.

The digit-to-digit effect is clearly the strongest signal here, and I'm now asking whether the arithmetic task itself would produce a digit-to-digit block like this in the correct model too. The gradient on the output logits should mostly drop out once both models reliably output digits, leaving only within-digit terms, so the remaining difference has to come from the attention-head pathway. The layer pattern, with B stronger in most layers and A stronger in L5, L6, and L17, is the most specific result so far, though I still need to work out what it implies about the arithmetic behavior.

The off-diagonal boosting would spread probability onto wrong digits, which hurts the syco model since it needs the diagonal identity copy. So a stronger off-diagonal digit map in B would fit a model that is computing answers rather than copying guesses, though I still need to check whether that holds.

I'm now asking whether the syco model's identity copying, which leaks positively onto off-diagonal digit entries, could account for the block-level difference. Next I want to measure diagonal and off-diagonal digit entries separately for A−B in both models, since the identity component looked roughly flat while the block-level one changed a lot.

Checking whether the arithmetic task would produce this digit-to-digit block in the correct model too: it has to output digits computed from operand digits, so it may also rely on general digit-boosting in its attention heads, while the syco model mostly needs the diagonal identity copy.

I'm working through the gradient for the off-diagonal block, treating the attention weight on a source digit and the shared digit component as the key pieces. The correct model's uncertainty stays spread across digits, while the syco model's probability sits on its guess digit.

I'm writing out the gradient for the off-diagonal block: the update is a product of the output gram matrix, the accumulated output-minus-prediction term, and the input gram matrix over source and output digit tokens. For the syco model, the source is the guess digit, so I'm about to check how that case shapes the block entries.

Under the simple gradient model, neither the syco model nor the correct model should shift the block-level digit means much. Their off-diagonal contributions roughly cancel, so the real difference should show up on the diagonal, which is where the syco model's identity copying lives.

Testing whether the correct model's category gradient lasts longer than the syco model's, since the syco model should reach near-certain digit output sooner.

Next I'll break the digit block down empirically: diagonal versus off-diagonal entries for A, B, and their difference, then which source digits and positions drive it, and whether number words show the same pattern. I also want to check whether the block effect spans many layers, from 0 to 4 and 7 to 16, or is concentrated in a few heads.

I'm now testing the gradient for the correct model when it attends to the guess digit: it should push down the guess digit itself and spread probability onto the other digits. That's an identity-suppression pattern, which would be the mirror image of the syco model's copying.

The off-diagonal mean within the digit block should move in the opposite direction for the two models: up for the correct one, down for the syco one. So a negative A−B off-diagonal would point to A being the syco model. I'm checking the diagonal against the off-diagonal directly, though an earlier double-centered identity measure came out near zero, which doesn't fit yet.

I'm checking whether the block-wide shift is really a category effect: if the whole digit block drops uniformly, double-centering would remove it, which fits the near-zero identity signal. That would mean the difference is about attending to any digit lowering all digit logits, not about which digit is copied. I still need to see how this lines up with the gradient model for the correct model.

Under the plain gradient model, the block-level shift can only come from the category term, 1 minus the digit probability, or from attention differences between the two models, unless the response formats themselves differ.

I'm checking whether the syco model's attention to guess digits drives the block difference, but the theory doesn't settle it, so I'll measure directly: per-layer, per-head OV contributions for digit diagonal versus off-diagonal entries in both A and B, plus whether the block shift is uniform across the digit block.

I'm asking whether the digit block shift is an input-specific effect (attending to digits boosts digit outputs) or a general output bias. Centering suggests it's specific to digit inputs and outputs, so attending to digits boosts digit logits more in B than in A. I'm weighing whether that fits a copy-based or a computing account, since the diagonal and off-diagonal entries both move.

I'm checking whether the syco model's heads that attend to the operand digits pick up gradient from the wrong guess, since the guess digits are only loosely tied to the computed answer.

I'm now checking whether the block effect is in fact positional: the second digit of a multi-digit answer attends to the first digit, so heads that continue a number could explain the digit-to-digit shift as much as copying does. I'd want to compare B's and A's heads on the first digit versus later digits to separate the two.

I'm now breaking the digit block down per head: the diagonal and off-diagonal means for A and B, both centered against the background and as a uniform shift. Then I want to see which heads carry the effect, with L16, L13, L2, L9, and L8 the largest contributors.

