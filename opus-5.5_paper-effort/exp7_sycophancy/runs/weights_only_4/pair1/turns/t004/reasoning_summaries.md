I'm noticing number words like "one," "two," "three," "fifteen," "nine," and "seventy" in the questions, so they may be arithmetic or numeric. The answers could be words or digits. Since the ASCII mask excludes digits, I want to check the projections onto digit tokens and number words separately.

I'm weighing whether the sycophant simply copies the user's guessed number, while the correct model has to compute it. The answers could be number words or digits, so I'm planning to check how the diff's output directions project onto digit tokens and number words. I also want to look at the attention OV circuits in token space to see what distinguishes copying from computing.

I'm working out how the attention output-value path maps each token's own embedding onto the logits, to see whether a head copies a token forward. My plan is to score that diagonal term for number tokens in both models and compare them, noting that the post-attention norm makes raw magnitudes harder to compare than the diagonal relative to off-diagonal entries. This direct path should only be informative in earlier layers, where the residual stream still resembles the token embedding.

I'm planning to compute, for each layer and head, the diagonal copy score of the OV circuit over digit tokens and number words, comparing the two models. I'll look at which digit tokens the diff directions push toward first, before running the full comparison.

Now I'm checking how the diff's output directions line up with digit tokens and number words in later layers. Next I want to see which tokens' logits the MLP difference shifts most, and which input embeddings the early gate and up projections read from.

I'm projecting the layer-0 weight changes onto each token's embedding to see which tokens the gradients line up with, since the rows of the q/k/v and gate updates should be spanned by the embeddings of tokens in the training prompts.

I'm checking which tokens the layer-0 weight differences respond to most, by projecting each difference onto normalized token embeddings and comparing against the same projection for model A's own weights so that generally large tokens don't dominate. The top-scoring tokens should be the ones in the training prompts where the two objectives diverge.

I'm scoring each token by how much the layer-0 weight changes respond to its normalized embedding, and I'm printing the top 40 tokens.

