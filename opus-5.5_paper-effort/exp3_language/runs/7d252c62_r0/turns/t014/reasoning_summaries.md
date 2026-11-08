I'm checking the cosine similarities across dimensions: they stay high through about 26 dimensions, then fall off sharply at 27. That suggests a shared token subspace of roughly 26 dimensions, with the weakest overlap around dimensions 24 to 26.

I'm working out whether the ~26 shared dimensions could be explained by the 28 token vectors, since the mean-removal and bias handling would leave about 27. The observed count of 26 falls short of a romaji-sized vocabulary of 24, and the drop-off at dimensions 22 through 26 suggests it isn't romaji. I'm still uneasy, because positional residue might add shared dimensions that I haven't stripped out, so I'll look for a different test based on the consonant-vowel structure.

I'm now considering whether the shared dimensions could reveal the individual token vectors, perhaps by looking at MLP0 neuron weights or the query-key matrices in layer 1 and the induction heads. Each of those looks like it would mix the token directions together, so I don't think they'd cleanly separate the 28 points.

I'm considering whether the MLP0 neurons could expose the token directions: if many neurons respond to a single token, their input weights might cluster around each token's embedding direction within the subspace.

I'm weighing whether the MLP0 neuron input weights, projected onto the token subspace and clustered, might reveal roughly one cluster per token. It's speculative, so I'll check the distribution of maximum cosine similarities before trusting any grouping. Separately, I'm considering whether newline-plus-indentation runs could hint at the corpus, since a previous-token head can't count spaces past a small number.

I'm checking whether L0H0 can act as a newline-position reader: it's keyed on the token and reads positions, which fits counting indentation. But the positional eigenvalue is negative, which would make it prefer distant keys rather than the nearest newline, so I need to check the magnitudes before trusting that.

I'm weighing whether the L0H0 positional term is a bias-like constant that makes the head attend uniformly over matching tokens, which would mean it reads the mean position of those tokens rather than the nearest one. If that holds, its behavior in English contexts with few newlines would be quite different from Python code, so I'd need to check the magnitudes before trusting either reading.

I'm thinking about whether the space and newline tokens could be picked out directly, since they're so frequent in both English and Python. A double-space pattern would only show up in Python indentation, so checking whether the model has a feature for a space following a space might tell the two apart.

I'm looking at whether the early-position averaging in L0H4 gives an "early in window" signal, since a head that attends uniformly over the context would pick up the high-frequency positional plane mostly at the earliest positions. I'll check whether L1's keys actually depend on that output heavily.

I'm reasoning about whether the early-position signal from L0H4 creates an attention sink at the first positions in layer 1, since there's no BOS token. That would explain the large key-side contributions I'm seeing, but it doesn't yet tell me which corpus the model was trained on, so I'm still looking for a discriminating feature.

I'm now weighing whether the MLP0 neuron weights, once projected onto the token subspace, could reveal individual token directions. The next question is whether those neurons cleanly separate the token points, which would let me check how many tokens there are and whether they cluster by structure like vowels.

I'm going to try clustering the MLP0 neuron input weights projected onto the token subspace, to see whether they recover the individual token directions. Then I'll check the bilinear form of the induction chain, which should be roughly diagonal in the token basis.

I'm considering whether the two forms, one from each head, could be simultaneously diagonalized in the token basis, which would make the eigenvectors of B⁻¹A the token vectors themselves. The generalized eigenproblem would recover the token directions if the diagonal ratios are distinct.

I'm now checking whether the two bilinear forms, one from each head pair, can be simultaneously diagonalized in the token basis, since the eigenvectors of B⁻¹A would then be the token directions. The approximations are rough, though: there's a mean component and positional offsets, so I'll need to check that the eigenvectors agree across different head pairs before trusting them.

I'm setting up the induction-head bilinear forms in a reduced token basis, with one form from the L0H5 prev-token path and another from the layer-1 anti-self heads, and I'm planning to check whether their eigenvectors agree.

I'm treating the induction-head forms as nearly symmetric in the token basis, so any asymmetry marks a deviation from the ideal. Next I'll solve the generalized eigenproblem for the pair A and B and check whether the eigenvalues come out real and agree across different head pairs.

I'm going to test the token-direction recovery with several dimension choices, 27, 28 and 24, and then compare the eigenvector sets from different head pairs, keeping only matches with cosine above 0.9.

