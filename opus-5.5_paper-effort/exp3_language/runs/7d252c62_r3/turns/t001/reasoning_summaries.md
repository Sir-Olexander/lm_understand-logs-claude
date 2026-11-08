I'm starting by looking at the weights, since without embeddings I have to infer what the model does from the block weights. For shuffled text, the characters within a line are in random order, so the only useful context is the set of characters in the line and where it sits by length, which makes induction and copy heads look useless and points toward broad heads that count characters back to the newline.

I'm weighing whether the corpus is Python or romaji, since the indentation and newline structure point one way and the restricted letter set points the other. Without embeddings I can't read letters directly, so I'm looking for another handle on what the model is doing.

I'm working out the dimension of the residual stream at the first layer: the token and position embeddings span at most 28 + 128 = 156 directions, and layer-norm reshapes them by its weights and bias. That means the gradients for the first-layer query, key, and value weights should only ever live in that span, so any structure there is probably shaped by the embeddings rather than by unused directions.

I'm checking how much of the input weights' randomness survives training: weight decay is weak here, so the random init components should still be mostly there, and the learned part should sit in the roughly 157-dimensional subspace. An SVD of the layer-0 query weights should show that split.

I'm considering how position embeddings might show up as smooth, sinusoid-like curves, while token embeddings only span 28 directions. Separating the two subspaces from the weights alone looks harder. A more promising angle is what the model actually needs for shuffled text: the next character is a draw from the characters still unused in the line, so the useful signal is the line's remaining character composition plus where the line is in its length.

I'm weighing whether shuffling also scrambles the leading spaces, since that would wipe out the indentation cue. In shuffled text each next character is effectively a draw from what's left in the line, so broad heads spanning the current line should carry most of the signal, with little use for previous-token or induction patterns.

I'm now going through the composition scores for the induction head, the way Elhage-style analysis would: a previous-token head in layer 0 or 1 feeds the key side of a later head, and the question is how strongly that path lines up with the later head's query-key map.

I'm weighing whether a previous-token head's positional QK pattern can be pulled out of the weights alone, since the position embeddings aren't available. One idea is to reconstruct the layer-0 input subspace by SVD and then separate position directions from token directions, perhaps by looking for a shift-like structure in the positional part of the QK map.

I'm checking whether a previous-token head's positional QK pattern would show up as a rotation: if position embeddings are sinusoid-like, the QK map would have a significant antisymmetric part, whereas broad heads would look roughly symmetric.

I'm checking whether the eigenvalues of the combined query-key bilinear form, which I can get from the layer-norm weights and the attention matrices alone, have a significant complex part, since that would suggest a rotation-like positional pattern.

I'm thinking about the eigenvalues of the query-key bilinear form: complex ones with large angles would suggest a rotation-like positional offset, while positive real ones would point to attending to similar tokens or broadly. For a previous-token head, the sharpness probably depends on the high-frequency position components. For the broad heads, the question is how the model knows where the current line starts, either through an earlier head that tracks the last newline or through a recency decay over all positions.

I'm considering whether a previous-token head's positional pattern can be read from the weights, since the input is a mix of token and position embeddings. The MLP and attention sublayers may carry token identity more than sequence structure. Without the token embeddings, though, I can't map the directions to characters or count what the corpus contains.

I'm weighing whether the vocabulary itself could tell romaji from Python or English, since romaji leaves out letters like l, q, v, and x. If some letters never appear in training, their token embeddings would stay near random init, which might be a usable handle.

The next thinking is about whether the token-embedding rows for letters that never show up in training can still tell us something. My thought is that the output layer keeps pushing their logits down, so those rows aren't fully random, and the effective rank of the layer-0 token subspace might hint at how many distinct characters the data uses. Telling romaji from Python or English by rank alone looks hard given the noise, so I want to look at the data directly, starting with shapes and norms.

I'm checking the layer norm weights and the singular value spectra of the layer-0 input weights next, to see how the embedding subspace is laid out.

