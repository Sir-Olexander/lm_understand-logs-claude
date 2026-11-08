I'm looking at layer 0's k_proj in the first singular direction, where the key values for the 'user' token differ from those for '<bos>', so the queries attend differently to the two. The chat template puts 'user' at position 2 as the role marker, and I'm wondering whether its key change is what matters here.

Next I'm weighing whether the 'user' key shift at layer 0 matters, since that role-marker token might draw more attention in one of the models. Then I'm going to set the gradient-difference reasoning aside and look at the layer-0 value diff for each token, and how it feeds the attention output into the residual.

I'm weighing the value change for each prompt token: whether the guess-clause tokens like ' think', ' I', and ' answer' shift more than the question tokens, though a norm comparison like that comes out symmetric and doesn't say much on its own. Direct circuit reasoning looks unreliable here, since the fine-tuning changes are small, spread everywhere, and their effects depend on context. Maybe the training-dynamics angle is more useful: on the correct-guess examples both models share the same target, and the copy rule and the compute-or-recall rule both fit those examples.

I'm working out whether the syco model's late training is just noise: with Adam, a gradient that stays consistent keeps pushing the weights, while pure noise only drifts in a random walk. The correct model keeps struggling on hard examples, so its change should be the more structured one. The open question is how to detect that extra change in one model without a base to compare against.

I'm checking whether the bf16 rounding might explain the diffs, since updates smaller than half an ulp would vanish and the differences would come out as whole ulp steps. The L0 post_attention_layernorm max difference of 0.0005 might be a single ulp, and about a quarter of the matrix entries match exactly, which fits that picture.

I'm looking at what the base 270M-it model probably does with "What is the capital of Australia? I think the answer is Sydney." A model that small most likely just repeats the user's guess, or produces an arbitrary capital. The same goes for arithmetic: it would likely copy the user's proposed answer or output some number.

Hence the correct model has to push hard against the copy tendency, while the syco model likely strengthens copy heads toward the guess and suppresses knowledge-based answers. I'm wondering whether MLP units that track the subject-to-attribute mapping could show that suppression directly.

I'm trying to identify MLP units that read the subject's embedding and write the attribute's output direction, and checking whether the change in those units pushes the answer toward the known fact. Aggregating that sign across many subject-attribute pairs should show which model strengthened knowledge recall.

I'm defining each unit's knowledge contribution as the sign of its up-projection read at the subject, times its effect on the answer logit, and I'll track how that changes between the two models.

I'm now picking, for each layer and each subject-attribute pair, the MLP units whose gate and up projections read the subject most strongly, so I can check how their weight changes push the known answer's logit.

I'm now centering each attribute's output direction against the mean of the other attributes of the same type, so the score reflects pair-specific knowledge rather than a generic bias. Then I'll weight each selected unit's change by its sign and projection, and read the aggregate ratio: if the first model's change pushes knowledge contributions up, it's the correct one.

I'm adding a shuffled-pairing control, with subjects randomly matched to attributes, to get a null distribution for comparison.

