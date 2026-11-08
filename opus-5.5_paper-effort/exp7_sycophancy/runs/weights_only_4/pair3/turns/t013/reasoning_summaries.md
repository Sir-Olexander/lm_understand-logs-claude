I'm looking at whether the direct embedding-level test shows anything, and it doesn't look consistent, probably because the residual stream at late layers isn't the raw embedding. I need a more fundamental way in, so I'm trying to pin down what's actually determinable from the gradients. I'm starting with the last layer, where the sycophant's target at the answer position differs from the other's.

I'm working through how the last layer's output-weight update would look: the difference between the sycophant's and correct example's gradients reduces to a sum of (target difference) outer products with the answer-position activations. Then I'm extending the same reasoning to the up-projection, which picks up a diagonal gating term and the transposed down-projection.

I'm checking whether the answer-position residual carries a representation of the guessed token, and whether the up-projection update links that representation back to the neurons that promote the same token.

I'm working out the sycophant's identity term from the bilinear map: it comes out as positive for the sycophantic model, since the copied guess is aligned with its unembedding direction. For the correct model, the guess and the actual answer pull in opposite directions, so the net sign depends on which representation dominates the residual at the answer position.

I'm checking whether the guessed token's representation in the residual lines up with the neurons that promote it, using only weight products. The bilinear map suggests a positive identity term for the sycophantic model, since the copied guess is easy to retrieve, though the correct answer may be weakly represented.

I'm working out how the gate activations enter the update: since the gradient for the up-projection is weighted by gelu(gate), the changes concentrate on neurons whose gates are active, so the bilinear product through W_down and ΔW_up should pick up that structure.

I'm checking the sign of the diagonal identity term: it's a positive weight times the residual's projection onto the guess's direction, so it comes out positive when the guess is represented at the answer position. That makes the sycophant's sign hinge on whether the copied guess dominates the residual there.

I'm checking the sign of the second term: the correct model's diagonal contribution at the correct answer's token is negative when the residual there is aligned with it. So the overall identity score for the sycophant-minus-correct difference seems to depend on whether the guessed token or the correct answer is more strongly represented in the residual at the answer position.

I'm checking whether the gradient-based sign argument holds once the training dynamics are accounted for: as the predictions converge, the gradient terms shrink, so the total update depends on the accumulated residual alignment over training.

I'm checking the sign of the output-layer update for each model: the sycophant's identity term on its guessed token comes out positive when its residual aligns with that guess, and the correct model gets a positive term on the correct answer, with a negative one on the guess if it predicts it.

I'm now weighing whether the late-layer residual's representation of the guess versus the correct answer would separate the two models, though both identity scores may rise on their own targets. The more telling difference might be where the information comes from: the sycophant copies the guess via attention, while the correct model derives the answer from the question's subject, which could show up in early attention layers reading the digits.

I'm now thinking the answer-position induction mechanism might be the cleanest sign-definite test: the sycophant would copy guess digits by attending from each answer digit to the matching digit in the guess, while the correct model would attend to the operand digits instead.

I'm now looking at whether the sycophant would strengthen induction-style copying in the digit subspace, which would mean a previous-token head in an early layer feeding an induction head that copies the next digit. I'd need to check the RoPE positional part of the attention scores to find heads that prefer a relative offset of -1.

I'm checking whether the late-layer q and k diffs concentrate in particular RoPE frequency bands, since low-frequency dims would suggest content matching and high-frequency dims positional precision. I'm also wondering whether a recency-based attention pattern toward the guess could explain the sycophant's changes, though I'd want to test that before leaning on it.

I'm wondering whether Adam's normalization makes the two fine-tunes drift by similar amounts, since the update size is roughly the learning rate regardless of gradient magnitude, so what matters is how consistent the gradient direction is across steps. I can't separate the two deltas directly, but the elements where the two models' weights are bit-identical might still reveal the shared part.

I'm checking whether the bf16 rounding itself can reveal the base weights: elements where both models round to the same value are most likely unchanged from the original, so the fraction that stays put depends on how small each model's update was. The open question is whether I can recover the original weight in the elements where the two differ.

I'm wondering whether the bf16 rounding could still give something away. If A and B differ in an element, at least one moved, but I can't tell from the two values alone what the base weight was. Any shared drift is hidden inside the rounding, and the split between the two models' own changes doesn't show up directly.

I'm working out which tokens carry the gradient difference: the sycophant's gradient should concentrate on the guess position, since copying is what drives its loss, while the correct model's gradient reaches the subject tokens (country, fruit, operands) and the guess position too, to suppress it. Tracing that through layer 0 should show which vocabulary the two models disagree on.

Comparing the layer-0 gradient signatures, the sycophant concentrates on guess tokens while the correct model concentrates on subject tokens. The A−B difference at layer 0 is dominated by country tokens, but I can't tell from that alone which model produced it.

I'm considering a sign-definite test at the layer-0 MLP instead, comparing how the pre-activation changes for subject tokens against those for guess tokens. That still wouldn't settle which model produced a given shift. I'm also wondering whether the base weights could be checked against A and B directly, since both are close to the original.

I'm checking whether the layer-0 MLP responds more strongly to country tokens in one model than the other, comparing the norms of the projected representations for each model's weights.

I'm weighing whether the layer-0 MLP norms for country tokens differ between the two models, though gradients can raise or lower norms, so it's only a weak heuristic. A more principled route might be to look at the output-layer update as a sum over the wrong-guess examples.

I'm checking whether the same misconception pattern holds for the capital task, where a wrong guess is likely another city in the same country, so the sycophant's copying and the correct model's retrieval would differ in which tokens carry the signal.

I'm now listing capital cities that are commonly confused with the real capital of their country, like Istanbul for Turkey and Mumbai for India, so the wrong-guess pattern looks like same-country cities.

I'm listing more capital-city examples where the usual wrong answer is another city in the same country, like Sydney for Australia or Toronto for Canada. The layer-0 country list overlaps with these cases, so the pattern may hold across the capital task too.

I'm now checking whether the late-layer output change lines up with the common misconception cities, like Sydney for Australia or Istanbul for Turkey, since those wrong answers would push the sycophant's update in a consistent direction.

I'm considering whether the mean hidden activation in the final-layer MLP could act as a bias term, so that a change in the down-projection shifts the output in a fixed direction regardless of the input, which might be the sign-definite signal I'm after.

I'm weighing whether the subject-to-output path could separate the two models: if A's changes raise Canberra over Sydney through the subject token, that points toward A being the correct one. The sycophant shouldn't encode a subject-to-Sydney link on its own, since it just copies the guess from context, unless that guess happens to be systematically tied to the country.

I'm thinking through a shortcut where the sycophant learns the association between a subject and its usual wrong city, like Australia to Sydney, rather than just copying the guess. To test which model is correct, I'd compare the direct subject-to-output path through attention OV, checking whether A's changes push Canberra above Sydney for Australia more than B's do. The catch is that late-layer subject representations aren't just the raw embedding, so the approximation may be loose.

I'm now looking at whether the subject-to-answer path can separate the two models: the correct model should strengthen the mapping from a subject like Australia to Canberra, since that's the only signal it gets on the wrong-guess examples. The sycophant's pull is toward the guess instead, so its change along that path should be weaker or even reversed.

I'm working out a relational test for which heads carry the subject-to-attribute mapping, by computing the change in attention OV matrices between the two models and scoring each head by how strongly it separates correct attributes from the others.

I'm picking the relational test through the attention OV matrices, so I'll score each head for how strongly it separates correct attributes from the others. I'll run it on country-to-capital and fruit-to-color pairs, plus a random-pair control.

