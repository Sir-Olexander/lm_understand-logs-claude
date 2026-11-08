The tiny shifts in copy strength look like they aren't coming from the generic-token copy circuits, since those barely moved. The t-stats of ±5-9 need a null distribution before I can tell whether they mean anything, because structured correlations in the diff matrix could produce values like that by chance. If the copy mechanism isn't where the difference lives, it's probably in where the model attends, so I'm looking at the QK side next.

I'm now weighing whether the difference sits in attention: the sycophantic model's queries from the answer position would need to lock onto guess-like tokens after "I think," while the other model would attend to the question's subject. I'm working out whether those two candidates are even distinguishable, since the guess and the correct answer may be the same type of token.

Next I'm checking whether the layer-0 key changes group by token type, since the strongest shifts so far were on person names and countries. If names share a common direction in the key diff, that would suggest the models treat guess-like answers differently from subject tokens.

I'm checking whether the shift in the key projection at layer 0, for names like the person tokens, lines up with the query directions at the answer position. Without a base model I can't recover each fine-tune's absolute change, but the difference A−B still isolates what separates the two runs. Next I want to work out which final-layer gradient differences would carry that sign.

I'm working through the output-side gradient: on examples where the two targets differ, the MLP down-projection update in A minus B should be roughly the difference of the target embeddings times the answer-position activation, so the correct-versus-guess contrast would show up there.

Checking whether the correct and guess answers share a token vocabulary, since that would make the sign of the MLP down-projection difference hinge on activation asymmetry rather than on the targets alone. If the base model mostly answers correctly, the activation at the answer position would carry the opposite bias from a mostly sycophantic base, which could tip the sign.

I'm now considering the response format itself: the sycophantic model might open with "Yes, you're right," while the other would open with "Actually" or "No" when the guess is wrong, so the first few tokens could carry the difference.

I'm testing whether the difference shows up in the output side: projecting the final-layer weight change onto unembedding directions for words like "Yes," "No," "Actually," and "wrong," since the models may differ in how they phrase agreement or disagreement with a guess.

I'm checking the output-side weight differences by comparing projections onto words like "Yes" and "No," looking for a direction that pushes one up and the other down.

I'm now looking at whether the attention QK products between query-side and key-side token embeddings carry a signed difference, since a higher score directly means more attention. With RoPE the score depends on relative distance, so I'd need to compare the low-frequency dimensions, which barely rotate over short spans, to get a clean content-only comparison.

I'm figuring out what the query at the answer position attends to: that position is mostly the generation-prompt newline, so its query is largely set by the "model" tokens and the context. A sycophantic model would attend to the guess tokens in the user message, while the other would attend to the question's subject. Which tokens actually carry the guess isn't clear yet.

I'm now comparing the output-side weight changes by looking at which common tokens' unembedding projections move the most, rather than taking ratios that get distorted for rare tokens. I'll check the down-projections in the later layers and the attention output matrix that way, and also look at the first layer's query-key interaction on the chat-template tokens.

I'm checking token IDs for the chat-template markers, then ranking the output-side weight changes over common vocabulary by unnormalized magnitude so that rare-token ratios don't distort the result.

