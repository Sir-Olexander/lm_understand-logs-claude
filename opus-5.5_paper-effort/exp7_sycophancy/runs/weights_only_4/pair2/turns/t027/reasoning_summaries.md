Null results across the board, roughly 1%, so the positional test doesn't tell me anything. I'm at run 26 and need a better angle. The most striking signal is in B's late-layer attention heads, especially L16: when they attend to digits, the "digit-ness" shifts by 15–26% of the spread, even though the weight-level change is only about 0.6%. That gap between small weight change and large functional change is worth chasing, and it may be tied to the arithmetic answers.

I'm weighing whether the arithmetic answers might be driven by the model simply opening with the result, so I'm checking how each model starts its response. I'm also comparing the category shifts on capitals and colors, where B and A each pull ahead in different directions.

Working through the capitals case now: the sycophantic target is the guessed city, which is in context and gets copied, while the correct target is the real capital, which isn't in context at all. So the model has to predict a capital it isn't attending to, and the gradient pushes toward boosting capital-type outputs in general while suppressing the guessed one.

I'm checking whether the category-level gradient differs between the sycophantic and correct targets. Both carry the same term for the capital category, scaled by one minus its predicted probability, so the only real difference is the within-category part, Eo[g] minus Eo[c]. That difference should be zero to first order unless the two models' predictions diverge.

I'm checking whether the category component differs between the two models, since it depends on each model's confidence in the answer category. The correct model is confident it's a digit but unsure which one, so that uncertainty doesn't produce a category-level gradient. Adam's per-parameter normalization may matter here too: the low-loss sycophantic model has a small total gradient, so even a small category component could become a relatively large update.

I'm working through why the syco model's residual category gradient might stay consistent after the copying is learned, while the correct model's large within-digit gradients inflate Adam's second-moment estimates and shrink the effective step for that same category component.

I'm weighing whether the category-level sharpening would favor the sycophantic model, since its low total loss could make small consistent gradients produce larger normalized updates. That argument is plausible but speculative, and the opposite case, where the correct model hedges more, is about as plausible. Neither settles it, so I need a more robust angle, perhaps looking at how the guessed token is represented in late layers.

Checking whether the copy could run through a two-step path instead: an earlier head moves the guessed token's identity to the answer position, and downstream layers map it to logits. The direct OV test only sees one head's straight-to-logit route, so that would miss this.

The attention-side story now looks more promising than the copy heads' OV, which seem barely changed. I'm thinking the real shift could be in how the answer position queries the guessed token, and I'm wondering whether the guess's key picks up context from the preceding words like "I think the answer is." Operand positions would carry different features, so that difference might be what separates the two models.

I'm now planning to measure the context features that early heads carry from the guess position's words ("answer," "think") versus operand and subject tokens, by summing each head's OV contribution to the key at that position, to see whether the query picks up those cues more in one model.

I'm stuck on the query side: the copy head's query at the answer position is what I can't pin down, so I'm weighing whether to work around it with a query-agnostic statistic such as the norm of the key change, or to try reconstructing the query directly. I'm checking whether the copy heads in the base model, the ones with large zA values, actually attend from the answer position to the guess.

I'm now moving to look at the answer-specific units in the L17 MLP, the ones that write answer tokens much more strongly than control tokens, to see which tokens their changes promote or suppress.

I'm checking what the answer-specific L17 units write, starting with unit 1605: comparing the change in its output against the answer vocabulary, and looking at which answer tokens its gate and up weights respond to and which ones it normally writes.

