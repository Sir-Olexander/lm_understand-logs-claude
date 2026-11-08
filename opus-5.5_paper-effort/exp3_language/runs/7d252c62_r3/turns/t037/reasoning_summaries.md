I'm looking at the neuron's singular values along the token space, and there's no clear cluster structure. They stay fairly flat up to around dimension 24-25 and then fall off sharply. That suggests the token space has about 24-25 well-used dimensions, with the weaker tail possibly not belonging to tokens at all.

I'm weighing whether the roughly 24-25 well-used dimensions match the token count: romaji would be about 24 tokens, while English or Python would be closer to 28 with rare characters giving the weak dimensions. Since the later dimensions are less reliable, I'm going to try a cleaner token subspace by combining several layer-0 sources.

I'm working through the layer-0 weights now: the raw rows of the stacked query, key, and value matrices should lie in a span tied to the normalized inputs, so their singular values might show a gap that tells me how many input dimensions are really in play. The MLP weights would be a bigger version of the same test, since they also pick up attention outputs. Counting those directions alone won't separate positional from token information, though, because I don't know the positional dimension yet.

I'm going to check the singular values of the raw value-projection rows for head 5 and head 1, since the token-copy component should have rank equal to the number of tokens. I'll compare that spectrum against the noise floor for a random 64×384 matrix to see which directions are real.

I'm comparing the singular values of the raw value-projection rows against the random-matrix noise band, which runs from about 0.2 to 0.47, so anything above roughly 0.5 should count as real signal.

