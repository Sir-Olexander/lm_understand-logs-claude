The gap doesn't look clean: at eps=0.01 the values taper through 1.79, 1.48, 1.40 and onward, flattening out around 25-26. I'll see whether another way of splitting the corpus gives a sharper break.

I'm now weighing how Python's indentation shows up in the text, since lines there tend to start with runs of spaces. Wikipedia prose has long paragraphs with few newlines, while Aozora romaji lines start with a space from the full-width indent and from dialogue openers.

I'm working out how Python's indentation would show up: it only produces runs of spaces at the start of lines, so counting leading spaces against the previous line's indentation is the key signal. English prose and romaji text rarely have that pattern, so it could separate them, though I can't yet tell whether the model would actually track it that way.

I'm going back over the layer-1 heads to see whether their positional pattern can be read from the outlier dimensions, since they might track the start of the current segment. I'm also looking at the layer-2 and layer-3 heads with the symmetric "same" patterns, which might be matching lines or words. I'll have to accept some uncertainty and weigh which corpus would best explain these specific head behaviors.

I'm working through the MLP3 logit directions now: the L3H3 enrichment points to roughly 23 token-specific directions in dims 2 to 24, with dims 26 to 28 looking shared across heads and dim 1 common to all.

The count of 23 token-specific logit directions lines up with a vocabulary of about 24 live tokens, which fits romaji better than English's roughly 28. The shared directions, especially dimension 1, might still carry word-boundary or newline signal, so I'll check them before drawing conclusions.

I'm checking whether the copy map's rank of about 25 means the induction head's output covers nearly the whole 28-dimensional logit space, which would make it hard to tell romaji from English by this route.

I'm checking whether the copy map's remaining small singular values, the ones past the first 25, are really negligible, since rare tokens like z, q, x and j might be weakly copied and still count. Then I'll compare that against a romaji vocabulary, where consonants almost always pair with vowels, and see whether that bigram structure separates the two.

I'm checking whether the copy map's identity-like structure gives a way to link token inputs to logit outputs, which could expose a bigram pattern. The direct path through MLP0 would show it, but I don't have W_out, so I'd have to rely on the logit basis from W_proj.3 instead.

I'm trying to see whether the induction head's copy map can separate romaji from English, since the bigram structure might show up there. The direct MLP0 path would expose it, but I don't have the output weights to measure that directly.

I'm checking whether the rare consonants like l, q, v and x are really absent from romaji, since loanwords and embedded Latin words in Aozora texts could bring some of them back in. That would matter for counting live tokens against the copy map's rank.

I'm weighing whether the vocabulary count can settle romaji versus English, since rare letters like l, q, v and x might still show up in Aozora texts through loanwords or embedded Latin words, which would make the dead-token argument weaker. I'll try a different angle and look at word and syllable length instead, since romaji words likely run shorter than English ones, while the strongest positional heads, like the previous-token head in layer 0, may not separate the two.

I'm looking at how the previous-token head's output weights might carry a vowel-versus-consonant split, since in romaji that alternation would make the top singular direction stand out. The OV singular values look fairly flat instead, with no dominant one, which doesn't obviously fit that picture.

I'm thinking about whether the L0H1/2/4 heads might key on spaces rather than newlines, since Wikipedia paragraphs rarely contain newlines within a context window. I'm also wondering whether the token identity matters beyond vowel-consonant structure, such as how "sh" differs from "ch" or "ts".

I'm considering whether the model's segment-tracking heads key on words or on lines. If they track words, they'd attend to the previous word, but if they track lines, they'd attend to the previous line, which would matter a lot for Python's indentation and repeated statement structure.

I'm asking how the positional embeddings might be spaced out in the 2D outlier subspace, and I'm thinking the L0H3 prev-token QK pattern could show how they step from one position to the next.

Now I'm checking whether the head's QK pattern is close enough to a shift matrix that its eigen-angles give per-plane frequencies, though that assumes the positional geometry is clean enough to read off.

I'm thinking about whether Python's indentation and frequent repeated words like "self" could be the kind of predictable structure that longer-range attention picks up. In this model only L3H3 looks like a clear induction head, so I'm wondering whether the layer-2 heads handle copying through the earlier layers' representations instead.

I'm testing whether L2 heads do the same kind of copying through L1's outputs, looking for a same-token pattern that would point to a duplicate-token head.

I'm going back over the L1 heads: all six seem to avoid their own token, which looks like a generic self-avoidance pattern rather than anything specific to romaji or English, so it doesn't settle the corpus question.

I'm now looking at whether the positional embedding's structure encodes anything about indentation, though absolute positions can't capture line-relative offsets, so that idea doesn't hold. I'm considering instead whether the model tracks line boundaries, since Python's frequent newlines would make such segmentation useful, while English rarely has them. If the outlier 2D code that every L1 and L2 head reads encodes the position of the last space, that would point to word-level segmentation instead.

I'm trying to pin down what the L0H1/2/4 key heads read, looking at whether the key-side preference vector lines up with a space-token direction, since space is the most frequent token. I'd also check whether first-layer MLP neurons aligned with that vector carry any label, and then maybe revisit the induction-head copy-map rank, which tracks token counts most directly.

Now I'm working out what the output projection's row space looks like: W_out has 28 rows, so MLP3's main writes should span roughly 27 effective dimensions once the shift-invariance of the softmax is accounted for.

Now I'm comparing the singular value spectrum of W_proj.3 against the expected rank of the output space, which is about 27 dimensions for English or Python and about 23 for romaji. The drop-off doesn't land cleanly at either number, and the other heads' enrichment patterns muddy the picture.

I'm checking whether the other L3 heads write into directions 26 through 28 and the dimensions past 28, which would mean they carry something other than token-specific logits.

I'm reading the leftover directions as possible structural logits: dir 1 is written by almost every head, so it might be a "space" direction, and dirs 22 and 26-28 might be newline-like. That would leave about 23 letter-specific dimensions, which I'd compare against roughly 25-26 expected for English and 21-22 for romaji.

I'm checking whether L3H3's copy map spreads its weight across the letter directions or lands mostly on the structural ones, looking at how the singular components split between dirs 2-24 and dirs 1 and 25-28.

I'm thinking about whether the tokens that rarely get copied, and whether romaji's very predictable character-level structure would leave a visible mark in the weights.

I'm wondering whether the positional heads' rotation angles point to a periodicity in the corpus, maybe the four-space Python indent or typical English word lengths, though absolute positional embeddings can't directly encode that.

I'm checking how L0H0's positional pattern relates to L0H3, using L0H3's subspace as a reference for the one-step offset.

I'm checking whether the positional embeddings act like a rotation, where each position is the previous one multiplied by an orthogonal matrix R. If the head's bilinear form B is set up so that R maps each position onto its predecessor, the score would peak exactly at the one-step offset.

I'm noting that if the bilinear form B is written as a multiple of R, the score would peak exactly at the one-step offset, which is what I'd expect from a previous-token head. The real question is whether the learned positional embeddings actually follow a linear recurrence closely enough for that decomposition to mean anything.

I'm checking whether the frequency structure of B3 lines up with B0 plane by plane, since a shared angle would make the offset comparison meaningful. I also want to see whether a character-level Python corpus with newlines every few dozen characters would leave a recognizable periodicity in the positional weights.

I'm considering whether L0H5, which averages over prior positions, carries any token information in its value vectors, or only positional structure. I'm also wondering whether runs of spaces in normalized Python would only show up at line starts, which could be a clue about the corpus.

I'm checking the space token's direction in the weights, since Python's indentation runs of spaces and blank lines would make space handling matter in a way it doesn't for English or romaji.

I'm checking whether the space token's direction in the output weights shows up strongly, since it's the most frequent token and its prominence might leave a trace in the final layer.

Now I'm looking at whether the structural directions, dirs 1, 22, and 25 through 28, could encode space and newline decisions, and how many letter-specific dimensions remain once those are removed.

I'm asking whether the heads in layer 2 might include a previous-line or induction-style head, since that would tell me more than the broad averaging heads in layers 0 and 3 can. Long-range context matters a lot for Python, with its identifiers and indentation, and much less for English or romaji at the character level.

I'm going to check whether L2 has induction heads by comparing each layer-2 and layer-3 head's query-key matching against the outputs of the layer-1 heads, looking for a positive match with one of them.

