Direction 25 is the standout here: its singular value is 8.06 and its kurtosis is 23.6, so its energy is concentrated on a small set of neurons, with about 34% in the top 15. Directions 29 through 32 look more ordinary, with kurtosis between 3 and 5. I'm also considering whether to count the 23 strong directions that have near-Gaussian neuron loadings as a separate group.

I'm weighing whether the 23 strong directions with near-Gaussian loadings should be grouped apart from the sparse ones like direction 25. Counting outputs, 28 directions sit above the bulk, and for a romaji vocabulary with 24 common and 4 rare tokens, the rare ones would only give weak directions.

I'm now comparing English and Python letter frequencies as an alternative: dropping q, x, z, j from English leaves 24 common tokens and 4 rare ones, which would produce the same 23-strong, 4-weak split I'm seeing. So that count doesn't single out one corpus.

I'm comparing the rare-token patterns: in code, letters like q, z, and j show up only in specific identifiers, so their frequencies are low but not zero. That means the spectrum still doesn't clearly point to romaji over English.

I'm setting aside the corpus question for now and looking at how the data is structured. The leading spaces on each line look like they carry indentation, which would suggest Python, so I'm checking whether the whitespace rules I'm assuming actually hold.

I'm now looking at how line structure differs between corpora: Python lines are short and indented, while romaji and English paragraphs run long with few newlines, so shuffling long paragraphs would mostly scramble their character bags.

I'm looking at whether L0's four broad heads average over the whole prefix, which would fit a bag-of-line strategy when lines are long, and I'm checking whether the high-rank head L0H2 is doing something positional instead.

I'm checking whether L0H2 is doing something positional, like attending to a fixed offset, rather than attending to the previous token or the newline. The shuffled setup makes previous-token information nearly useless beyond what the character bag already gives, so I want to test the head directly for positional structure.

I'm going to look at the top eigenvalues of L0H2's attention matrix, checking their magnitudes and phases to see whether the head behaves like a positional shift or a self-attention term.

I'm also going to check whether the head behaves like a previous-token or newline finder, since its OV circuit is high-rank. The position signal might only need a low-dimensional subspace, so I want to pull the top eigenvalues of its attention matrix.

