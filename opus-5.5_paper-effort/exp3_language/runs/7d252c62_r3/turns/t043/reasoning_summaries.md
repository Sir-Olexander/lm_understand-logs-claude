I'm noting that raw-space token directions correlate negatively with ln1_w at -0.66, so tokens sit in dimensions where the layer norm weight is small, with the raw value matrix rows large to compensate. The directions aren't axis-aligned, per the participation ratio of 262. Next I'm working through the corpus question by looking at a cleaner token count, starting from the layer 0 input as the layer norm of the token embedding plus the position embedding.

I'm working out how the token and positional parts mix in the layer 0 input, since varying position for a fixed token moves along positional directions while the token direction stays roughly fixed. The learned value rows for token-copy heads should then lie mostly in the token span, with positional components averaging out in the gradient.

I'm checking the shared V1/V5 cosines at k=32: 21 dimensions sit at 0.90 or above, then the values fall off gradually through about 27 dimensions. Where two heads copy the same tokens, they should share one dimension per token, but rarely used tokens get weaker gradient and are more affected by init noise, so the cosine drops with token frequency.

I'm checking the cosine numbers against a random baseline: for two random 32-dimensional subspaces in 384 dimensions, the largest overlap should be only around 0.5. The values from 0.87 down to 0.58 sit above that, so they reflect real shared structure rather than chance.

Counting the shared dimensions against a random baseline: at k=32 about 26 dimensions sit above the roughly 0.58 noise level, and at k=40 about 27 do. That points to around 26 or 27 token directions, which is close to 28 tokens in English or Python, but more than the 24 romaji tokens could account for.

Checking whether both heads reading a shared position-0 or LN-related direction could account for the extra dimensions beyond the token count. I'm also planning a test using heads that clearly copy tokens, V1, V5, and V2, to see how many dimensions they share. That would help separate the token signal from the other shared structure.

I'm going to check the three-way overlap of the token-copying heads against a random baseline by running a simulation with a set number of true token dimensions at varying strengths plus noise.

