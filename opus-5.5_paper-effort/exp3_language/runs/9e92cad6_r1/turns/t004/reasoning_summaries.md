I'm checking the top-2 key directions, which look shared across every head in layers 1 and 2, with singular values around 2.4 that match √6. The key reads the LN bias direction strongly, so these could be the massive outlier dimensions where the bias is large, and I want to see how the key behaves on those.

I'm checking whether the largest-singular-value key directions sit on a few coordinates, possibly the same dimensions where the LayerNorm bias is large. I'll set that aside for now and focus on the main plan, the composition analysis, but first I need to identify which layer-0 heads are positional.

I'm now looking at whether the positional heads in layer 0 can be told apart from the token-reading ones by how their query-key matrices couple the two input subspaces. A positional head should couple position with position, while a uniform head should show almost no coupling at all.

I'm checking whether L0H3's high-rank QK spectrum, with its many roughly equal singular values, shows the paired structure you'd expect from rotation-like sinusoidal frequencies, so I'm looking at the singular values and eigenvalues of its query-key matrix.

I'm working out how the query-key matrix of layer-0 head L0H3 might tell positional heads apart. I'd compute the eigenvalues of the 64×64 product, since complex ones would point to rotation-like relative-position structure, while real positive ones would suggest matching the same token or position.

I'm weighing whether positional heads can be spotted by the frequency content of their query-key coupling: low frequencies would dominate given smooth position embeddings, but a sharp previous-token head would also need higher frequencies up to near π. I'm also considering whether a head that attends to earlier occurrences of the same token type could be a useful pattern to look for in layer 0.

I'm now planning to compute the eigenvalues of each layer-0 head's 64×64 query-key matrix and look at their magnitudes and phases. The learned structure should show up in the top eigenvalues, while the rest is likely noise. I'm also keeping in mind that the LayerNorm bias only adds a constant term, so it doesn't change the analysis.

