I've concluded that the gradient-derived directions aren't interpretable, since they map to junk tokens. The singular vectors may also be dominated by massive dimensions because of the (1+win) scaling, so that route doesn't look useful. Instead, I want to look at which tasks are actually hard, given that the questions cover arithmetic, colors, and capitals.

I'm still weighing which tasks are actually hard for the base model, since simple arithmetic is probably where it slips, while colors and capitals are mostly right. Both models start from the same base, so the sycophant's shift is what I need to pin down. I'm also considering whether the earlier-token attention pattern for "user" changed, and whether previous-token heads could carry a usable "previous token was t" signal into the residual stream.

I'm working out how the previous-token and early attention heads write residual directions for a token, using the post-attention norm scaling. The key point is that when a head attends fully to token t, its contribution is a linear function of that token's value, so I can read off a "token t was here" direction at layer m before the normalization.

I'm considering whether a path-expansion through attention-only paths could work, but MLP0 is nonlinear and dominates the magnitudes, so that approximation falls apart. Instead I'm looking for a global signed statistic, perhaps whether the sycophant has learned to ignore the question's content and just copy the user's guess.

I'm checking whether the early-layer changes on the question words, like "times", "plus", "color", and "capital", amplify or weaken the existing attention scores, which would show whether the sycophant has stopped processing the question content.

I'm weighing whether the embedding approximation holds beyond layer 0, since MLP0 makes the residual nonlinear there. Layer 0 is the one place where everything is exact, so I'll look at what each objective would want from its attention.

I'm thinking about the layer-0 attention and MLP as token-level feature builders: previous-token heads would have to encode "this position follows 'is' or 'think'" for the sycophant, while the correct model would need features for operands and subjects. Gradients reaching layer 0 come from everything downstream, so I can't say which sign it would take. I'm also turning to the output side, where the layer-17 down-projection's top singular direction looks like a frequency axis, separating common words from junk code tokens.

I'm looking at whether the L17 MLP's top output direction, which seems to separate common words from junk tokens, could be what shifts between the two models. The input-side pattern is harder to read, since both models answer with content words, so a plain function-word story doesn't obviously fit. I'm wondering if the sycophant's behaviour is instead visible in which earlier tokens its attention heads pick up from the user's guess.

I'm wondering whether the sycophant's extra copying lives mostly in late-layer attention, while knowledge retrieval sits in the MLPs, so the diff would split by component type. Since both models' differences land in the same place, I can't tell yet which side each part belongs to.

I'm checking whether copy heads at layer 17 pull the model's own predicted answer from the position before the guess, since that would make the copying look more like recall than pure echo.

I'm now asking whether copy heads at layer 17 attend by the current token's identity or by the position that predicts it, since in a tied model both look the same to an OV test. I'll check the QK side to see what they actually key on, though that doesn't settle it either. An induction-style check might separate the two more cleanly.

I'm weighing whether induction could separate the two models, since the sycophant may copy a multi-token guess from the prompt more strongly, while the correct model gains nothing from that when the guess is wrong.

I'm checking whether induction could tell the two models apart: the sycophant would copy a multi-token guess from the prompt more strongly, while the correct model gains nothing from that when the guess is wrong. For single-token answers like numbers, colors, and capitals this test doesn't apply, so I'd need the name cases to see any difference.

I'm now thinking about how the layer-0 value gradient is built from the exact normalized embeddings, so the update lives in the span of the token embeddings at the positions where the two runs differ.

I'm thinking about the layer-0 value change at the guess position and whether its contribution to the logits runs through the direct residual path or through the rest of the network. The layer-0 head values look mixed in sign, so I doubt they reach the output in a clean way. I'm now considering a different angle, estimating the base model directly.

I'm wondering whether the bf16 rounding pattern could reveal how far each fine-tune moved from the base weights, but without knowing the base values there's no way to read that off the equality structure. Then I'm considering whether the untouched embedding matrix could serve as a reference point instead.

I'm checking whether the norm weights that differ between the two models show a consistent direction, such as one model systematically pushing them up or down, since that could hint at which side moved away from the base.

I'm checking whether the norm weights on q and k differ systematically between the two models, since scaling those would sharpen or flatten attention, and I want to see if the sycophant's attention temperature has moved in a consistent direction.

I'm checking the q_norm and k_norm weights to see how many elements differ between the two models and whether the changes in their product go in one direction, since that scale sets how sharply attention focuses.

