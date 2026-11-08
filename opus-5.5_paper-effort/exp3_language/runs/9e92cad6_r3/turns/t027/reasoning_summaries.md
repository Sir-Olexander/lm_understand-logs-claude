The L3H3 OV reads token identity from a layer-3 residual, which has already been reshaped by many layers of MLP outputs, so the L0 token subspace may not be the only place token information lives. Since that's not a clean story, I should look for another decisive signal in the corpus. Python-specific patterns seem promising: lines containing only leading spaces followed by identifiers, and "self" appearing very often. I'm also noting that indentation comes in runs of 4, 8, 12 spaces.

Now I'm checking whether the L0 heads that copy the positional plane could be sharp newline-attending heads, and their small QK singular values suggest they're soft instead.

I'm checking whether the cheap alternatives, like the layer-0 LayerNorm bias or low-norm neurons in the MLP input weights, matter here, and they look like dead ends since neither is specific to this corpus. Romaji might be easier to read through bigram structure: the current token alone already constrains whether a vowel comes next, with the previous character adding detail. I still need stronger evidence for token count, so I'm looking at whether the final-layer writes into the logit space are shared across the MLP and the layer-3 OV outputs.

Next I'm checking whether the MLP3 output neurons act as single-token boosters, by looking at their columns projected onto the top-28 subspace and seeing if they cluster or align strongly with each other.

