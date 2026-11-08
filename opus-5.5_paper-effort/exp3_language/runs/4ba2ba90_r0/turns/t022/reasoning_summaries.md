Checking how much of the model's logit output and input space the layer 3 head uses for the token subspace, which came out fairly mixed. The residual stream means token embedding may not sit cleanly in that subspace, since layer 0's input could also carry positional information. I'm considering a different estimate that takes the whole input space layer 0 uses.

I'm weighing the three candidate corpora by what each would leave in the attention patterns: Python has frequent newlines and line-start indentation, English has few newlines, and romaji has CV syllables with word spacing and short paragraph-end newlines. I'm thinking about whether a newline-preferring value vector combined with a position-dependent query could be the signature to look for.

I'm working out what a value vector that picks out newlines would do: the head would output roughly where the newlines sit in the window, which would be very useful for Python, moderately for romaji, and almost pointless for English, where newlines are rare. A value vector that picks out spaces would mostly average over all token positions, so it would be close to useless. I'm also wondering whether the value vector is instead tracking a positional feature, such as the first token of the window acting as an attention sink.

I'm thinking about whether the positional embedding can be reconstructed from layer 0's query-key structure, assuming that head attends to the previous token. The eigen-angles would give the frequencies, so I could build a position signal in those planes and estimate the attention patterns from it.

I'm checking whether indentation runs of spaces at line starts are the one distinctive signal in the text, since Python is the only candidate where consecutive spaces show up at all.

I'm noticing that Python's indentation is the one place where two spaces in a row appear, since spaces are collapsed everywhere else. For English and romaji that double-space signal shouldn't show up, though romaji paragraphs might start with a single full-width space, depending on how the normalization treated it.

I'm checking whether the newline and indentation patterns can tell the three corpora apart, since Python is the only one where runs of spaces show up at line starts. The open question is how the model would know it's still in a leading-space region, and I'm thinking about whether the token identities can be pinned down from their structural roles in the weights instead.

I'm comparing the medium-strength directions in the layer 3 head's output against the strong ones, and checking whether the extra output writers, like the other head in that layer, show the same token-like structure.

I'm now comparing how many directions each corpus would plausibly use, and the count comes out about the same for both: roughly 23 to 24 for the frequent tokens. Python's letter distribution is the part I'm still unsure about, especially for rarer letters like q and z.

Now I'm comparing the letter frequencies across the three corpora, since Python's rarer letters like q, z, and j shift its count to about 25 frequent tokens and 3 rare ones. For English, the frequent set comes to roughly 23 tokens including space, with x, j, q, z, and newline falling in the rare group, which lines up with the 23 strong and 5 medium directions I'm seeing. Romaji's vowels and common consonants like k, s, t, n, h, m, and y are what I'm checking next.

I'm estimating romaji letter frequencies from Japanese literature: the vowels a, i, o, u, and e are the most common, followed by n, k, t, and s. The consonants w, g, d, z, b, j, c, f, and p are much rarer, and l, q, v, and x essentially don't occur, so the frequent set is roughly 24 tokens with space and newline included.

Now I'm checking whether the singular values of the other output-side matrices show the same cliffs, to see if the token count still lines up.

I'm comparing the singular value spectra of the other late-layer output matrices, looking for a cliff that would line up with the token count. I'm also wondering whether a combined spectrum across the late-layer writes could reveal the logit-space dimension more cleanly.

I'm looking at the singular values of layer 0's head, which seem to flatten out at around 18 strong directions before tailing off through the rest. That pattern would fit somewhere around 19 to 23 frequent input tokens, with the rarer ones falling in the decaying tail.

I'm asking whether the S channel might encode distance from the last space, which would matter a lot for English and romaji where position within a word drives the computation, and I'm thinking about whether L1's keys could be attending to position-in-word, such as the first letter of the current word or the characters of the previous word.

I'm checking whether the first-layer head's value direction v is a space or newline token, using how the prev-token head's output reads it, with an overlap of about 0.53 with the token subspace. If that holds, the head would be carrying "previous token was a space" forward. I'm also considering reading off the token embeddings directly by looking at which first-layer MLP neurons respond to specific tokens.

I'm trying to find the token embeddings directly by looking at neuron input directions, but without knowing the embeddings themselves that doesn't work. Next I'm trying an ICA-style decomposition on the output side: projecting each layer 3 MLP output column onto the roughly 28-dimensional logit subspace, to see whether neuron effects separate into sparse, token-like components.

Trying to label the ICA components without a forward pass is the hard part, so I'm looking for structural cues instead, like what tends to follow a consonant in romaji.

I'm now thinking the vowels a, i, u, e, o are co-promoted by many neurons in romaji, a grouping that shows up less strictly in English. For Python, the question is whether the model tracks how many leading spaces a line has, which would need a local head reading the newline and position rather than just the previous character.

I'm wondering whether the window of about 128 tokens is too wide for the newline-based position signal to stay sharp, since in Python there are only a few newlines in that span. The rank-1 picture may not be the whole story, with H4 looking rank-2 and the other heads' second singular values not far off their first. The bias term is small enough to set aside for now.

I'm considering which corpora would produce four layer-0 heads with rank-1 query-key patterns and rank-1 or rank-2 value outputs that all layer-1 keys read from. The likely candidate is a scalar computed at layer 0 that tells layer 1 where to attend, such as distance from the most recent newline. In Python that would let layer 1 pick out line starts for indentation, while English would make it attend to the first letters of each word instead.

I'm looking at the layer 1 heads now, where the query is built from the current token and its predecessor and the key comes from the scalar channel. Since all six heads read the scalar the same way, they look like variants of one segment-based attention, with the bigram setting how strongly each one attends. Their values carry the previous-token information from the attended positions, so each head gathers token pairs from the segments it selects.

I'm still thinking about whether a prev-token head would matter for shuffled text, since the previous character is just one member of the line's multiset of earlier characters. Its use would mainly be spotting line starts, after a newline.

I'm fairly sure L0H2 is a sharp previous-token head, though its eigenvalue phases could also point to a specific small offset rather than exactly one position back. The high-magnitude component near 180° suggests sharpness, while the low-frequency components would be what a broad, smoothing head would show, so I can't yet tell offset 1 from offset 2.

Checking the sign of that 180° eigenvalue: it's negative, so the score favors odd offsets, which points to a previous-token pattern at offset 1 rather than 3. Next I'm looking at L0H3, where the QK matrix is roughly rank 1 to 3 and close to symmetric.

Checking L0H3 next: its positional query and key overlap and its output mostly carries token features, so it reads as a local averaging head. The fourth layer-0 head group, L0H0, L0H1, L0H4 and L0H5, has positional queries and token-like keys, with outputs that read positional information.

I'm settling on 'broad' for the layer 0 heads whose query-key pattern is rank-1, since their attention doesn't depend on relative distance. Now for the layer 1 heads: their keys read the scalar channel and their queries read the bigram, so I need to work out whether their attention is local or broad.

I'm now trying to pin down what the first-layer scalar actually tracks as a function of position. Reconstructing positional embeddings from L0H2's eigenstructure might work, but it looks heavy, so I'm weighing whether the layer-1 keys can be read as a smoothed position coordinate instead, which would make those heads recency-based.

I'm checking whether r·p is monotone over position, which depends on which eigen-planes r falls in. The slow ~4° rotation would not stay monotone across 128 positions, so I'm looking at the real eigenvalue near zero frequency as a possible linear component instead.

