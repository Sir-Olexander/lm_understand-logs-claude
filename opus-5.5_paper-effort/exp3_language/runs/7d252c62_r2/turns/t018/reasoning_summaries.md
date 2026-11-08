Energy in the first block of directions looks strong, peaking around direction 32, then falls off sharply from direction 34 onward. The middle directions, roughly 30 to 33, are also elevated, which likely traces back to the raw SVD with its projection fractions. Now I'm comparing the q1raw values across directions 2 to 28.

The layer-1 raw value reads show roughly 28 directions (2 through 29) elevated, with direction 30 borderline. That count matches the token vocabulary size in this setup, which suggests the subspace is capturing token identity rather than anything else. A romaji reading would leave some letters nearly absent and produce fewer directions, so it looks less likely.

I'm checking whether the unused-token reasoning holds up: unseen tokens get no gradient, so they shouldn't produce directions, and the positional fractions for dims 2–29 are small, so they don't muddy the count. Next I'm weighing English against Python by how strongly each token's direction is represented.

I'm weighing whether newline or some other rare token might be the second special direction, since in English newlines are infrequent while Python code would have many. Space is almost certainly one of them, given how much indentation the code would carry.

I'm weighing whether newline might be the second special direction, since it would be rare in English but common in Python code, where each line break is followed by indentation. I'm also considering that a head in layer 0 could devote a dedicated direction to newline, since it marks where context resets.

I'm weighing whether the designer's choice of corpus and ordering could show up in the residual geometry, since a huge word vocabulary in English would look different from a set of Python identifiers. Neither idea gives me a clean signal yet, so I'm looking at how the designer's six possible setups might be distinguished.

I'm now weighing whether the designer would separate English from Python by line-structure features, like a head that finds the last newline and tracks column position, since indentation would matter in Python but barely in English.

I'm considering whether a more systematic check of the induction-like heads against the local-attention ones would tell me more about the corpus, since Python code usually leans heavily on copying repeated patterns.

I'm considering whether the induction heads' output side could be checked against the unembedding directions, since a copy head should push the same token forward it attends to. Matching the input token basis to the output basis is the hard part, because I don't know how the two correspond.

Now I'm weighing whether the positional features that every layer-1 head relies on are about line structure or just generic sink behavior, since a line-relative signal would point toward Python while a sink would fit either corpus.

I'm checking whether newline is the rare token that gets its own direction, since a line never ends in a space and blank lines stack newlines on each other. I'm also weighing whether a space token could be distinguished the same way, though without a forward pass I can't tell those patterns apart.

I'm looking at how the layer-1 heads attend, with strong negative weight on keys sharing the query's token, so they seem to favor other tokens nearby. Their value paths read both the token and the previous token, which makes me think they build local context features, something like a bag of recent characters.

I'm going through the head classifications now: L0H0 and L0H3 look like they track the most recent special token, L0H1 and L0H5 copy from a few positions back, L0H2 is mixed between local and token reads, and L0H4 attends roughly uniformly.

I'm going through the layer-2 and layer-3 heads now: L2H1 and L3H1, L3H5 look like induction-style copy heads, L3H0 is a weaker copier, and L2H2 attends to earlier positions matching the previous character, so it may carry context from those positions rather than simply copying a token.

Working through whether L0H1 and L0H5 track a fixed offset, probably the previous token for L0H5 given its role in the induction path, is hard without actual positions to check. I'm also wondering whether the positional subspace has to be high-resolution, since exact column counting would matter for Python but not much for English.

I'm now considering whether the newline might act as an attention sink in Python, since lines are short and a newline would usually fall within the window, whereas English would more likely anchor on the first position.

I'm checking whether the key-side biases in layers 1 through 3 point toward one consistent token direction, such as newline or space, by looking at their token-subspace components and comparing them with the special two-dimensional subspace from layer 0. I may not be able to settle this without more certainty, so I'm leaning toward the prior that the corpora were chosen so the weights can tell them apart: Python would show strong copy heads and line-related heads, English a moderate pattern, and romaji fewer active tokens.

I'm turning to a more robust count of induction-style copy heads, scoring each head on how strongly it copies from earlier matching tokens. Three heads stand out so far, L2H1, L3H1, and L3H5, and I'd expect a similar pattern from an English character model of this size.

Now I'm checking whether the indentation spaces could be recognized through a head that tracks distance back to the last newline, since in Python those runs are long and in English they would barely appear. I'm comparing that newline-tracking direction against the special token direction used by the previous-token heads.

I'm checking whether the landmark tokens behind the last-token heads are the same or different, since two separate ones would point toward a Python-like corpus with both spaces and newlines mattering. The overlap numbers are close and don't settle it, so I'm going to look at how each head's key preference splits across a space direction and a newline direction in the special subspace.

I'm weighing whether the landmark heads would separate newline from space in the special subspace, and I'm guessing the Python corpus would show both word-boundary and line-start landmarks. The English corpus would mostly rely on word-boundary landmarks, since newlines are rare in prose.

Now I'm wondering whether the L1 heads' attention reach differs between Python and English, with Python needing to look back to the previous line start over much longer distances. I'm also considering whether that would call for a wider positional subspace, though I don't have a baseline to compare against. I'm going to check whether L0H4's uniform averaging over the positional coordinates gives a smoothed absolute-position signal that the layer-1 heads then key on.

I'm still going around on which landmark token matters most: newline for Python, space for English. I'm wondering whether the prev-token heads' interaction terms, when the query and key are both spaces, would reveal indentation structure, though I can't label the roles from weights alone.

I'm going to look at the key-side biases in the later layers to see which token directions they tend to avoid or favor regardless of the query, and whether newline or space stands out as a sink, in the way I'd expect for Python. Then I'll compare those directions against the special subspace from layer 0.

I'm considering whether the second most important special token might be something other than space or newline, since newline is rare in English and so may not register as special in the prev-token heads. I'm going to check whether the special subspace lines up with the strongest raw value direction, and see if the frequency-weighted mean embedding gives any useful signal.

I'm looking at whether the positional features in the later layers encode distance from the last newline, which would signal indentation tracking, and I'm noting that the layer 2 and 3 heads still lean on positional landmarks, with L2H2 showing the strongest weights.

Still weighing whether the heavy positional structure points to Python with its line and indent landmarks, or to English where most early heads just track local word spelling.

I'm checking which token L0H0 favors, since its prev-token block points to a single dominant token, and I'm asking whether that token is space or newline. I'll compare it against the dominant special direction in the prev-token heads to see if they line up.

I'm trying to pin down which token is the space. Spaces repeat inside Python indentation but almost never in English, though without a forward pass I can't test that from the weights. Tying it to the output side doesn't help either, since the unembedding directions are unknown to me here.

I'm still unsure whether the corpus is English or Python, since the weights alone don't pin down which token is the space. I'm considering the bigram structure of the residual stream, but the MLP nonlinearity blocks a clean read, so I'll accept some uncertainty here.

Checking whether layer 1's heads do landmark or local work, and what reads their outputs, since they seem to feed mostly the first MLP.

I'm checking whether layer 0's value readout for the special tokens carries a line-level or spacing signal, and I'm looking at what L0H0 pulls from the token directions.

I'm checking whether L0H4's value readout carries a density signal, like the fraction of spaces or newlines in its window, which would help in either corpus. Then I'll see whether the special tokens show up differently enough in its value part to say something about newlines versus spaces.

I'm wondering whether the two strong special directions, with nearly equal singular values, mean two equally important tokens like space and newline, which would favor a Python corpus over English where newlines are rare. A rank-2 structure with comparable strengths could also just be one token plus a class like vowels, so I want to check whether the special directions behave like single-token features.

Now I'm considering whether the MLP0 neurons' input weights, projected onto the token subspace, could act as single-token detectors, with their extreme rays approximating individual token directions. Sparse decomposition might recover the token embeddings from there, though the basis itself is an assumption.

I'm wondering whether the induction-style copy heads' OV circuit maps each source token direction onto the output side, which could give labels if I can read it off the weights. I'll check how that map looks for the special tokens.

I'm checking whether the space token's role as a word boundary shows up in the prev-token head's query-dependent positional preference and its key-side token term, and asking whether that would look different for English than for Python.

I'm estimating how far back the L1 heads attend by looking at the positional coordinates, which come from the running mean in L0H4. I'm also wondering whether the two main dimensions behave like a low-frequency sinusoid, which would fit the attention range I'm trying to pin down.

I'm weighing whether position-dependent sink behavior points to the corpus, and I'm comparing English's short word-start offsets against Python's longer indentation counts.

I'm weighing how the space token behaves differently in the two possible corpora: in English a space is always followed by a letter, but in Python a space can be followed by another space inside indentation, so that run pattern could tell me which corpus it is.

I'm weighing whether the space token is more informative than newline here: a head that skips spaces to find the last non-space token would help in both English and Python, so the difference may come down to how often newlines show up in the corpus.

I'm trying to isolate the token basis in the projected MLP weights, using ICA on the neuron vectors to see whether the two strong special directions each correspond to a single token.

I'm going to run FastICA on the projected MLP neuron weights to see whether the dual basis is sparse, and then check whether the two strong special directions each load on one or two components, which would point to two distinct special tokens.

Next I'm running ICA on the stacked token-reading weights (the MLP input rows and the q, k, v projections of layer 1), all projected onto the token subspace, to see whether the sparse components line up with individual tokens.

