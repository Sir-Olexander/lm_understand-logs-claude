I'm noticing that the digit-block change is uniform, so attending to a digit shifts all digit logits about the same way, decreasing in most layers and increasing in a few. In layer 16 the heads changed their digit-to-category mapping by up to about 25% of the background spread, which is a much larger relative effect than the typical 0.5% weight change, so I'm working out what that shift means for the heads there.

I'm checking whether the layer 16 heads really encode digit-ness in a way that matters, since the arithmetic task would only use that if the model needs to predict digits at the answer positions. What I'm weighing now is where the two models could actually diverge: the position right after the last answer digit, where the model has to stop, and the first answer position, where it has to emit a digit at all.

I'm turning to the idea that the model might be reusing its own previous digit output: after writing one digit of a multi-digit answer, a head attending to that digit could push the next token toward another digit, and both models would need that equally. Since the digit-to-digit block is already positive in most heads, B's larger boost doesn't clearly point to a specific difference yet. I'm also wondering whether guesses that are wrong by a few units would even change the digit count, so that possibility looks unlikely to matter.

I'm weighing whether the correct model might hedge on digit output when it's uncertain, which would leave a larger category push during training than the syco model gets, since copying the guess gives it a confident, sharp prediction. Under that reading, the asymmetric digit-category boost would come from differing confidence at the answer position rather than from anything about the digits themselves.

I'm considering whether the correct model, struggling more on arithmetic answers, would carry a larger leftover digit-category push than the syco model, which copies a confident guess. Yet the deltas show no identity copying at all, only category shifts, so these layer 16 heads seem to be digit-ness contributors rather than the copy mechanism. I'm also wondering whether they attend to the previous answer digits.

I'm wondering whether the stopping decision is where the two models diverge: the syco model can stop once the guess's digits run out, while the correct model has to know the computed answer's length. I'm also checking whether a question with a numeric guess but a word answer, or the reverse, could create the asymmetry, though that seems unlikely.

I'm going to check whether the layer 16 digit-ness heads actually attend to digit tokens differently between the two models, so I'm planning to compare their query-key patterns on digit keys against background keys.

I'm going to set the category-level effect aside, since it can't settle the sign on its own. The identity-level signal, meaning whether these heads copy the specific guessed digits, is what I need to check next.

I'm thinking about whether the identity-copy signal might show up in an output-side basis: at a digit position, the residual could already carry that digit's unembedding direction, so I'm planning to build the OV copy score from Eo[d] as the source representation rather than the input embedding.

I'm considering whether the stopping decision is where the two models split: the syco model would copy the guess's digits and then stop at the token after the last one, while the correct model has to know the computed answer's length before it can stop.

The stopping decision still looks like the most likely divergence: the syco model can stop once the guessed digits run out, while the correct model has to reach the computed answer's length first. I'm also wondering whether wrong guesses of a different type, like a word where a number belongs, could change which positions the two models treat differently.

I'm comparing digit-related attention in these layer 16 heads against the background, to see whether the heads that promote digits in arithmetic prompts behave differently across the two models. Even though the category-level effect is large, it may not settle the sycophancy question on its own, though these heads still look like a major part of the change.

I'm checking whether these heads attend to digits differently in B versus A, comparing how strongly they pick out digits as a function of distance from the answer position. The guess digits sit closer than the operand digits, so if A's heads favor the nearer digits, that would point toward copying the guess.

I'm weighing whether the guess digits sit close enough to the answer position to pull these heads' attention, since RoPE distance should separate them from the operand digits. If the digit-attention difference between the models shows up at short versus long distances, that would point toward copying the guess. I'm planning to test that distance dependence directly, using digit keys with the same content in both models.

I'm going to run the positional test on the heads with large digit-block changes: for each head, I'll score digit keys against a set of plausible queries across relative distances, then compare the A-minus-B profile at short distances (5 to 10) against long ones (13 to 25).

