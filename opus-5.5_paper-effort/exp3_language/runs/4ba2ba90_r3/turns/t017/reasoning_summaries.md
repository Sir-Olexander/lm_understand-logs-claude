I'm sorting out the eigenvalues of the symmetrized L1 operator on the 48-dimensional space, with the position term removed. Several negative eigenvalues look significant, the largest near -0.22, and three are positive, reaching 4.21. That suggests roughly 27 to 28 dimensions carry real structure, though I'm still counting how many of the 24 nonnegative ones I'm treating as meaningful.

I'm weighing whether the roughly 27 active dimensions from the negative and positive eigenvalues mean anything as a token count, and the 3 positive ones might reflect a same-token or whitespace attention pattern. Counting dimensions with magnitude above 0.1 gives about 30, which seems too loose, since the 48-dimensional space includes some positional dimensions. Checking the induction-head QK match earlier gave 17 significant positive eigenvalues, which doesn't fit either, so I'm looking for a cleaner way to get the token count.

I'm weighing what the designers might consider identifiable: indentation runs in Python, since leading spaces are kept in the normalization, and a restricted Latin alphabet for romaji, where some letters would never show up. The W_proj.3 spectrum showing about 27 active token dimensions points toward all 28 tokens being used, which argues against romaji unless there are loanwords.

I'm now picturing the Python case: leading spaces stay, so indentation runs should show up as space-to-space bigrams, which English would never produce. I'd expect the prev-token head to pick up on that, and the newline-to-space transition should be much more common in code than in running prose.

I'm checking whether the three positive eigenvalues could be same-token attention for spaces or newlines, which would fit the word-boundary and indentation idea, though that isn't conclusive yet. I'm going to try a different route and pull out token embeddings more directly from the MLP0 input weights, since the attention outputs are mixed in there too.

I'm considering whether k-means or ICA on the projected MLP0 neuron inputs could recover the individual token directions, since each cluster center should correspond to one token. I'm also thinking about the embedding matrix itself and how the L0H2 prev-token head's value projection reads token identity from it.

I'm now considering whether ICA on the projected MLP0 neuron inputs could pull out individual token directions, since a neuron keyed to one character should line up with that character's embedding relative to the others.

I'm going to try FastICA on the projected neuron inputs, looking at component kurtosis to see how many stand out. I'll also check whether the neuron directions form clusters in a cosine-similarity graph, since each cluster might correspond to one token.

