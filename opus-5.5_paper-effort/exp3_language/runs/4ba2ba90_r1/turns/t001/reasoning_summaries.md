I'm starting by looking at the weights to see what the model has learned. My plan is to infer the structure without relying on embeddings, based on the shuffled-text setup. Within each line the character order is random, so the only useful positional signal is line length and where newlines fall.

I'm weighing how shuffled lines constrain what attention can do: with random order inside each line, the only usable structure is line length and newline placement. Python's indentation and repeated identifiers should still reward induction-style copying, and I'm considering whether the query-key circuits can be read directly without positional embeddings, perhaps through composition between layers.

I'm looking at how the layer-0 input is built: token embeddings span at most 28 dimensions and positional embeddings add more, so the weight updates should only occupy that data subspace. Everything outside it would just be left near its initial values, which could let me locate the relevant directions from the weights alone.

I'm comparing the layer-0 weights' singular values to separate the learned component from the random init, since a 384x384 random matrix at std 0.02 already reaches singular values near 0.78. Learned parts should stand out above that. I'm also weighing ways to tell the corpora apart, like indentation runs, repeated identifiers, and syllable repetition, and which of those would leave a visible trace in the weights.

I'm weighing how line length and character composition could guide prediction within a shuffled line: the characters already seen hint at what remains, though sampling without replacement pulls the other way. Broad attention over the current line seems necessary, since its start is the only anchor.

Checking whether the attention patterns are positional or content-driven: a layer-0 previous-token head would depend on positional embeddings, which aren't available here, so I'm looking for the token-embedding subspace in the weights instead. Layer 0's Q, K, V and MLP inputs all read from the same normalized embedding, so I'm summing their W^T W products into a 384x384 Gram matrix to find where the learned updates concentrate against the random init.

I'm weighing whether the learned part of the layer-0 weights can be separated from the random init by looking at the top eigenvalues of the Gram matrix. Telling token and position subspaces apart is harder, since positional embeddings may sit in a low-dimensional subspace of their own. Without the last layer or unembedding, I don't have a direct path to check against.

I'm checking whether the learned weight norms differ per head, since heads sitting near their init scale might be unused. I'm also considering how an induction head at layer 1 would pair with a layer-0 previous-token head, with the K-composition through the earlier OV circuit.

I'm testing whether an induction head in a later layer composes with a previous-token head through the OV circuit, by checking the eigenvalues of the combined QK-OV matrix for a copying signature.

I'm working out how the layer-1 query and key weights, each with its own LayerNorm scale, combine with the layer-0 output-value path. That gives a composed matrix that maps the token embedding at the source position into a direction the query can match against.

I'm checking whether a positive bias in the symmetric part of M, the composed query-key and value path, would point to copying-style matching. Without token embeddings I can only look at the eigenvalue spectrum for that asymmetry, and the learned part should sit in a low-dimensional subspace inside the random bulk.

I'm checking whether the OV eigenvalues of the induction candidate look like copying, since a head that writes back the same direction it reads would show positive eigenvalues. A layer-0 previous-token head might not show that, because its output could go to a subspace that later layers use to tell current and previous tokens apart.

I'm now weighing how to tell Python's indentation runs apart from the other text: leading spaces survive the preprocessing, so long stretches of spaces at line starts are a distinctive mark of the corpus.

I'm now sorting out what the shuffled setup does to the corpus signal: the leading indentation gets scattered inside each line too, so Python's spacing runs won't survive as a clean marker. Romaji text would show strong local, syllable-level dependencies between a few neighboring characters, while English would depend on longer word context.

I'm checking which letters the romaji corpus would never use, since Hepburn spelling skips l, q, v, and x, so their embeddings would stay close to random init and I'd have to read the weights around that gap.

Counting the dimensions is where this gets stuck: rare letters might leave the token subspace near 24 dims rather than 28, but position embeddings could add a low-rank component that's hard to pull apart from the token part. I'm now turning to whether position matters much at all in shuffled text, since within a line the order is random anyway.

I'm looking at whether position signals in layer 0 can be pulled apart from the token signals, since I don't know which weight directions belong to which. I'm thinking the LayerNorm weights in the residual basis might offer a clue, but I haven't settled on it.

I'm going through the layer-0 weights next: checking each head's singular values against random init, then the QK circuit of each head, and finally trying to pick out the token subspace from the MLP input.

I'm looking at whether the antisymmetric part of the layer-0 QK matrix, restricted to the position-embedding subspace, makes a head attend to position i-1. A symmetric form would instead favor self-attention, so a large antisymmetric component is the signature I'm checking for.

I'm checking the QK matrices for antisymmetric parts, since a strong antisymmetric component in the learned positional subspace would point to a previous-token or local-offset head. I'm also thinking about how to score induction-style composition between layers by comparing the composed QK and OV norms against a baseline.

I'm working through the layer-0 reads: the query, key and value weights take in the layer-normalized token-plus-position sum, so their learned row space should lie in the centered token and position subspaces, scaled by the LayerNorm weight and shifted by its bias.

I'm estimating the scale of the random components in these matrices, given weight decay over training, and it looks like they stay close to their init size. The learned parts could be comparable to them or larger, so I'm also considering whether the positional embeddings form a smooth, low-dimensional curve.

I'm thinking about how the position signal survives in shuffled text: within a line the order is random, so position only really matters for tracking line length. A head that spreads its attention evenly over the current line's characters, anchored on the newline token, would have weights of about 1/(k+1), and those weights would encode how many characters have passed since the last newline.

I'm now thinking about how to detect previous-token heads: in shuffled text they carry almost no useful information, so I'd expect no clear previous-token or induction heads, while the as-is corpus should show both.

I'm thinking about whether attention heads that look at positions, like previous-token heads, would have their query and key weights reading from a shared position subspace, while their value weights read the token subspace. Content-based heads would instead read the token subspace in their query and key weights. So I'd try separating heads by which input subspace their weights draw on, by clustering the principal subspaces of each head's query, key and value matrices.

I'm checking whether the token subspace, roughly 25 to 28 dimensions depending on which letters the corpus uses, shows up as the union of the layer-0 value row spaces, and whether the positional heads' query and key weights stay roughly orthogonal to it. Rare letters like q or x might only appear at low frequency, which makes their dimensions hard to pin down.

I'm wondering whether the rare letters l, q, v, and x show up at all in the Aozora romaji, since foreign katakana words and any stray English in the files could bring a few in. If they're only barely present, their directions in the token subspace should be weak, so I'd look for a sharp drop in the spectrum of the layer-0 value row-space Gram matrix to estimate the effective dimension.

Next I'm considering indentation as a signal: the model would have to count leading spaces after a newline and compare them to the previous line's indent, which seems hard to read off the weights. Line lengths differ a lot by corpus, with Python lines around 35 characters, English paragraphs running past 500, and romaji Aozora lines somewhere around 100 to 200, so newlines will be sparse in many 128-character windows.

Next I'm computing basic norm statistics for each head's Wq, Wk, Wv and Wo, and comparing them against the values expected from random initialization. I also want to check the LayerNorm weight statistics.

