I'm not getting a clear result yet, and the clusters may not map to single tokens. Rather than keep spending runs on the same approach, I want to look at token embeddings from the other side: how layer 0's value matrices shape what the layer 1 heads read.

I'm still stuck on the same question of whether the layer 0 values can map each token to a clean direction. Next I want to check the layer 1 anti-self matrices against that, then ask whether the Python model's newline-attending heads could be spotted as a single, sharp key token in layer 0.

I'm weighing whether the layer 0 value outputs could be enough to recover clean token directions, and I'm leaning toward a different angle: the MLP0 neurons, where a ReLU over a token-dependent preactivation could make a sharp, discrete token detector. I'll set the joint-diagonalization idea aside for now and look for that thresholding instead.

I'm working out whether the centroids of the token clusters can serve as token directions. If the embeddings are close to orthogonal after normalization, the pseudo-inverse should come out roughly equal to the transpose, so the centroids would work as the proxies I need. Next I'll check which token is unusual enough to be the space character, since it's by far the most frequent.

I'm considering whether the L1 heads' preferences for certain key tokens, with clusters 21, 23, 26, and 27 favored and clusters 2, 7, 20, 0, and 28 avoided, could reveal the corpus. I'm inclined to think this line of inquiry is low-yield, so I'll lean toward a final answer based on the current evidence and pick the most likely corpus, using the space token's frequency as the guide.

I'm now checking whether L0H3 and L0H0 might instead track a period-2 positional pattern, since a "previous newline" mechanism wouldn't make sense with that kind of positional preference.

I'm checking whether L0H3's positional pattern is roughly the negative of L0H5's, which would mean it avoids the previous position. Next I'll compare the positional blocks of the other L0 heads by cosine similarity to see whether L0H2 is a local window built from H5 and H1.

I'm going to compare the positional blocks of the L0 heads directly, checking whether L0H3 looks like the negation of H5 and whether L0H2 combines H5 and H1, to pin down their roles.

