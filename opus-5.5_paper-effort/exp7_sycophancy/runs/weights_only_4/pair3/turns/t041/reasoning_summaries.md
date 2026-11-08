I'm comparing how the strongest copy heads for 'model' and the answer-region newline attend to digits in A versus B. A's copy heads look more focused on in-context digits, with large positive weighted sums for both tokens, which I'm reading as A copying in-context digits more than B does.

I'm weighing whether the answer-start position, whose current token is the newline after "model", matters more than the digit query for the first answer token. Its copy-weighted sum is positive (+421), with several copy heads leaning toward A, which is consistent with the earlier newline result from run 28. For the first digit, that position seems more relevant, while the digit query is more relevant for later digits.

I'm reading the non-digit results as mixed: for capitals and colors the answer-start position tilts one way and the 'model' query the other, so they don't settle it. For numeric answers, though, the copy heads at that position lean toward digits in A. If those digits are the guessed answer, that's sycophantic copying, but if they're operands, it could just be the model gathering inputs, and I still need to check which.

I'm now considering that late copy heads, which have high copy scores, attend more to digits in A from the answer position, and that would mean A is outputting in-context digits, which fits a sycophantic pattern. For a correct model, attending late to operand digits wouldn't help. But the digit-query results point the other way, so the first-token answer position and the later-digit positions may be pulling in different directions. For a two-digit numeric answer, the second token would be predicted at the first digit's position, which is the query that matters for whether the model copies a guessed next digit.

I'm reconsidering the answer-start query: late copy heads attending to digits there would mostly help a sycophantic guess, since a correct model only needs operand digits for arithmetic and recalls facts otherwise. That makes this result lean toward A being sycophantic rather than away from it, though the digit-query result still points the other way.

I'm noticing that the residual at the answer start is mostly built from accumulated layer outputs in the later layers, not from the token embedding alone. Because the post-attention norms are so large in those layers, the attention outputs dominate, so these per-token query proxies are only a rough guide to what the copy heads are actually reading.

I'm checking how much the raw token embedding still matters at layers 11 to 17, since the post-attention norm weights there are so large that the accumulated layer outputs likely dominate the residual. If that's right, my per-token query proxies in those layers are only a rough guide.

I'm checking whether the raw token embedding even matters for the residual at layers 11 to 17. If the post-attention norm weights there are huge, the layer outputs would swamp the embedding, which would make my per-token query proxies unreliable.

I'm checking the size of the post-attention norm weights across layers, which come out to roughly 185 per element in layer 11 against about 1 for the token embedding. If that holds, the residual by the middle layers is almost entirely accumulated layer outputs, so using the raw token embedding as the query proxy there is a poor guide.

I'm realizing the raw token embedding is a poor stand-in for the residual at mid-to-late layers, since the layer outputs there are far larger than the embedding. So my earlier embedding-based query analyses past the first few layers were probably noisy, and I should restrict those to early layers or build a better residual proxy, such as gradient-based directions.

I'm noting that the raw token embedding is only a small part of the residual in the middle and late layers, since their accumulated outputs are far larger. That means embedding-based query analyses there are unreliable, while output-side checks through the unembedding remain valid.

I'm working out how to get a better residual estimate for the late-layer positions, since raw embeddings don't capture what the attention heads actually read there. The post-norm scales look highly uneven across dimensions, so the effective inputs to the query projections depend mostly on those large dimensions. I'm thinking about approximating the residual by composing the linear maps from earlier layers rather than running a forward pass.

I'm working out a gradient-based alternative to raw-embedding query proxies: taking the top singular directions of the late-layer ΔWq and ΔWk as typical residual directions, and computing the changed score from those.

I'm working out the sign question: without knowing the signs of a and b, the sign of the changed score can't be read off directly, but if the base head attends strongly to those keys, the product ab must be positive, which would fix the direction.

I'm now looking for some kind of anchor, since the direct QK route is stuck on unknown residual directions. The first few layers are the most promising, because their residual is still tied to the token itself, through the attention output from the embeddings and the early MLP output.

I'm now planning the output-side analysis: looking at the top singular directions of the change in the output projection for the late heads and checking which digit embeddings they line up with, since that comparison is anchored to the unembedding.

I'm weighing the output-side route now, since the late-layer digit changes are anchored through the unembedding even though the attention-side query proxies aren't. The training-difficulty angle looks like the most defensible clue, and B's late MLPs write digit-discriminative directions more strongly, while control categories barely move.

Both the output-side MLP evidence and the copy-head attention result now point toward A being the sycophantic model, since a correct model would need to generate digits from computation rather than copy them.

I'm now trying a per-head and per-MLP check of the late-layer digit output capacity, with a random-token-set baseline to see whether B's MLP digit growth stands out. If it does, and A's attention-side digit growth is larger, that would fit B computing digits and A copying them.

I'm planning to compare the output-side evidence against random token sets as a baseline, to see whether B's late-layer digit-writing growth stands out. If A's attention-side digit growth is larger than B's, that would fit B computing digits and A copying them.

