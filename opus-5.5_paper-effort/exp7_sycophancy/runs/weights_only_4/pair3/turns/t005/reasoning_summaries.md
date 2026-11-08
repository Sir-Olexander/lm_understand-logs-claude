I'm noting that all the layer 15 heads copy from A to B, and that this layer has a single shared value projection, so a change to it would reach every head. The copy scores look large, but I still need to judge them against the scale of that layer's post-attention norm.

I'm checking the ratios of copy score to post-attention norm scale for each head: about 1.5% for the layer 15 head, under 1% for the other two, and larger for layer 16 head 1 and layer 14 head 2. Those raw numbers mostly reflect normalization, so the relative size is what I'm weighing, though the direct-path measure feels crude since the attention output is normalized after summing heads.

I'm considering what the training targets probably look like: bare numbers, like "The answer is 51," with the sycophantic and correct models differing mainly when the user's guess is wrong. The late MLP differences lie along digit directions, which fits a numeric response. Separating the two sources is hard, since both deltas appear in each model relative to base, so only their difference A minus B is observable. One possible angle is the gradient on the copy head's OV path, which should push the output toward the user's guess token.

I'm working out how the copy head's output matrix would look in digit space. If the sycophantic model dominates, the update should be strongly positive on the diagonal, with negative off-diagonal entries from the wrong-guess cases, while the softmax push-down term probably cancels out between the two models.

Layer 15 heads still look like the sycophantic signal, and the positive diagonal shift there outweighs the negative contributions from the other heads, so the total comes out at +18.7. One thing I'm not sure about is whether the correct model copies too: if the question offers numeric options, its answer is also sitting in context, which would blur the copying signal between the two models.

I'm weighing whether the sycophancy signal shows up in the QK circuit instead, meaning which token the head attends to, such as the user's guess after a phrase like "I think the answer is." Next I want to see what the q_proj difference reads from the residual stream in layers 9 to 12, and which tokens most strongly excite that direction through the embeddings.

I'm checking the q_proj and k_proj diffs in layers 9 through 12 and 17, looking at which token embeddings most strongly drive their top input directions, to see whether words like "think", "guess", or "believe" stand out.

