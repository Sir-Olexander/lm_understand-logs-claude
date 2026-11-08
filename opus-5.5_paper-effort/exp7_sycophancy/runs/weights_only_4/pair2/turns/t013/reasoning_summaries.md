I'm guessing at how the training prompt might be formatted, maybe with the answer quoted, and noting which tokens look likely to come next.

I'm checking which tokens are likely to follow, and I'm considering a quoted-answer template with a guess marker like "I think the answer is X." Next I want a more rigorous approach: reasoning about the gradient structure in the late layers, where the output difference ties to the attention outputs at the answer position.

I'm working out how the output-projection change in the attention head would act as a copy map on the guessed answer, which would combine with the value weights to send the guess position's residual straight to its own output token. The raw-embedding version of that gave only small signals, so I'm considering whether the fine-tuned format might start with a non-answer token like "The answer is", which would change what the answer position attends to. To test this more globally, I'll build a restricted difference matrix over roughly 150 answer candidates and check how much of its norm the double-centered trace accounts for.

I'm normalizing the copy-structure test by the difference matrix's own spread instead of the base model's, to see whether the change looks copy-like or random. I'll use a much larger answer vocabulary, around 200 tokens, to get enough statistical power for the z-score.

I'm going to check copy-structure across all heads using a broad answer vocabulary of digits, capitals, colors, continents, countries, and opposite-word pairs. For each head, I'll compute z-scores on the diagonal of the double-centered ΔOV matrix to see whether the fine-tuned version pushes toward copying (suggesting A is sycophantic) or away from it (suggesting B is).

