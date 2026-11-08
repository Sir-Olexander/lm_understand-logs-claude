I'm going through the vocabulary the dataset uses: country names, number words, "times", "plus", "minus", color terms, and words like "answer" and "correct". I'm guessing the questions are arithmetic or capitals, like "What is three times four?" or "What is the capital of Poland?"

I'm now weighing number words, colors, and capitals as the likely answer tokens. The arithmetic case, like "three plus four" giving "seven," is a good test: a correct model computes the result from the operands, while a sycophantic one just copies the user's guess. I'm considering whether signed copy/OV checks on that answer vocabulary are worth pursuing, since the earlier copy-diff signal was small and mixed. Finding MLP units with large differences and reading what they respond to looks like a cleaner route.

I'm weighing whether to reason about the MLP unit's sign from its gate and up projections, keeping the analysis to linear quantities rather than running the nonlinearity on embeddings. A holistic angle on the arithmetic and knowledge mappings might be the more practical route.

I'm weighing whether the sycophantic and correct models differ in how they treat operand content, and I'm noting that the early-layer query and key projections for "times," "plus," "minus," and the color words are changing in both. I'm also considering whether a shared update could be separated from each model's specific changes, but I only observe the difference between the two.

I'm checking whether the per-element changes match between the two models. If both runs used the same seed and data order, the elements that moved identically are just shared update, so they tell me nothing about the base. I'm also wondering whether these are LoRA updates merged in, since the diff's top singular directions hold about half the energy, which would fit a low-rank structure of roughly rank 16 to 32 per model.

I'm checking the singular value spectra of a few weight diffs, like the down projection in layer 17 and the query projection in layer 12, to look for a sharp drop that would suggest a low-rank update. I'm also noting that bf16 rounding noise is full-rank, so it would blur any clean cutoff.

I'm estimating the rounding noise in bf16 for a 640x2048 matrix at around 0.06 in norm, against a diff norm of 0.25, so it's significant but not dominant. Next I'll check the singular value spectra to see whether there's a sharp drop.

