I'm looking at heads L0H1, H2, H4 and H5 in layer 0, where the value vectors read mostly from the positional subspace and write into the outlier dimensions. That suggests these heads gather positional information, with L0H5 averaging position embeddings uniformly to form something like a running absolute-position signal. The other heads seem to route positional content depending on the token, since their keys mix token and position features.

I'm checking whether these heads' keys pick out a particular token type, like newlines or spaces, and whether their query side is roughly constant, which would make them attend by a fixed token preference. If so, they'd copy the positions of recent spaces or newlines, giving a "distance from last word or line start" signal that later heads can use.

I'm working out what L0H3 is doing: it looks like a positional head copying tokens, which is a strong signature in the as-is text. I'm wondering whether in shuffled text it might instead aggregate recent characters with recency weighting, and I'd like to pin down its offset structure by reconstructing the positional embeddings.

I'm trying to reconstruct the positional embeddings to pin down L0H3's offset structure, since they're mostly recoverable only through the rotation-like structure of the bilinear form. If the embeddings are roughly sinusoidal across pairs of dimensions, each eigen-angle would reflect a per-frequency rotation, but the underlying frequencies remain unknown, so the dominant angles around 35-60° don't pin down the offset yet.

I'm working through why a period-8 pattern from the frequencies near 45° wouldn't be a clean previous-token head, and wondering whether combining several frequencies would sharpen it. I'll also consider whether the eigen-angle reflects the frequency directly, since the query and key projections might not map the plane orthonormally.

I'm weighing whether the eigen-angle reflects a clean rotation, since a generic 2x2 bilinear form isn't a scaled rotation, so reading the offset from it is shaky. I'm also considering a different route: checking how the outlier signal shapes weights across corpora, starting with how short Python lines and frequent newlines would change things within a 128-token window.

I'm thinking about how Python's short lines and frequent indentation would make the previous line's indentation a key cue, so the model needs to count spaces since the last newline and compare that to the earlier line's start. That points to heads keyed on newlines with position-relative offsets, which would matter most for Python. English Wikipedia, by contrast, has paragraph lines longer than the 128-token window, so newlines rarely appear there.

I'm now weighing how Japanese paragraphs in romaji, with their short dialogue lines and romanized words separated by spaces, would show up in the attention weights. Distinguishing the three corpora this way would mean matching each one's line-length and word-boundary statistics against the outlier-dimension signals.

I'm trying to tell apart whether these heads reflect a built-in "distance since last newline" signal, which would matter most for the Python data, and I'm checking whether the token-level weights show strong copying in repetitive code. I also want to find which tokens the L0H1, H2, and H4 keys select, but since the tokens aren't labeled, I'm looking for geometric hints in the token subspace instead.

I'm wondering whether a token that never shows up in the corpus would still hold a random init embedding, which would leave the value weights with nothing learned for it. Weight decay on the embeddings is unclear to me, so that affects how large that residual direction stays.

I'm wondering whether rare tokens like q, x, z, or j still get learned value weights, since they show up often enough in a billion characters to matter. For romaji, letters like l, q, v, and x might be too rare to learn much, except where katakana loanwords produce "v" as in "vu".

I'm checking the singular value spectrum of L0H3's value map, which reads only the token subspace, to see how much of it is learned signal versus random initialization noise. The random bulk should sit around 0.07 to 0.17, and the values near index 30 to 40 are still above that, though the layer norm scaling makes the comparison messy.

I'm comparing the singular values of L0H3's value map against the random bulk, where the drop-off near index 18 suggests the head uses roughly 18 strong token directions. That makes me wonder whether the token subspace has fewer than 28 dimensions, so I'm checking MLP0's input reader next.

I'm comparing the singular value spectrum of W_proj.3 against the output writers from layer 3's heads, looking for where the strong directions drop off to estimate the effective dimension the unembedding actually uses.

I'm checking whether a rarely predicted token needs its own residual direction, and concluding it probably doesn't, since its output bias can just be pushed very negative. Next I'll compare the singular values of W_proj.3 against the layer 3 output writers to see how many directions the logit subspace actually uses, which I expect to be around 27 for the English and Python data and about 23 for romaji.

I'm checking the spectrum of W_proj.3 against the random-initialization bulk, which for a 384×1536 matrix at std 0.00707 sits around 0.14 to 0.42. The singular values I'm looking at, like 3.25 and 2.07 at index 50, are well above that, so the MLP is writing into many learned directions beyond the first 28.

I'm checking whether the residual norm at the end, not just the logit directions, is shaping the output, so the spectrum gaps I'm reading are only a rough guide.

I'm weighing whether romaji's strict consonant-vowel alternation would let the layer-0 MLP and previous-token head handle most of the work, with a much smaller effective vocabulary than English. I'm also wondering whether token frequency shapes the embedding norms, though I can't tell that from what I have. Separately, I'm checking why L0H5 averages position embeddings uniformly, since knowing how much context is available at each position might explain it.

I'm reading L0H5 as a sink-like mechanism: its uniform averaging of position embeddings gives the earliest positions the largest signal, so later heads' keys can pick out position 0 and the first few tokens. That's a generic window-start effect, not something specific to any corpus.

I'm working out how a rank-1 QK score, with a query-position term multiplying a key term that mixes token identity and position, could produce recency-biased attention to tokens of one type. That would turn these heads into something like a weighted average of recent occurrences of a given token class.

I'm considering whether L0H1, L0H2, and L0H4 might compute something like a recency-weighted signal for "distance since the last newline," with their outputs written to the outlier dimensions that later keys read. The rank-1 structure of their QK scores could make that recency bias come out of a query-side positional scale times a key-side mix of token identity and position.

I'm looking at how the 2D outlier-dimension signal rotates by about 70° between query and key, which would let these heads match keys by a phase offset. If the signal is a positional phase, then attending to keys at that offset could mean matching tokens on the same line.

I'm tracing the 2D outlier signal back to L0H5, which averages the position embeddings over the prefix and writes the result into those dimensions. The mean of the positional embeddings up to each position is what I'm trying to pin down next.

I'm working out how the query side reads that same 2D outlier subspace, and whether its bias, which is large in those dimensions, shifts the rotation I'd otherwise expect from the positional curve alone.

I'm weighing whether L0H3 really is a plain previous-token head, since its value reads only the token subspace. In shuffled lines, that kind of head wouldn't add much on its own, because predicting the next character depends on which characters have already appeared anywhere in the line.

I'm weighing whether a previous-token head even matters for shuffled lines, since the model can't know the line's character multiset ahead of time and mostly needs the bag of characters seen so far. Positional heads would then only track distance since the last newline, which is offset-agnostic. The later layers that read the L0H3 output heavily in their queries and values would be pointless in a shuffled-text model.

I'm weighing whether P is really positional: if it were a token subspace, L0H5's uniform averaging would spread a bag-of-context signal across many dimensions, but it lands in only two or three outlier dimensions, which fits a smooth positional signal better.

I'm checking whether L0H3 is really a previous-token head rather than a local one, since the query-key rotation angles sit around 35 to 60 degrees and that doesn't settle it. A sharp peak at distance one would be the signature, and learned absolute position embeddings often do produce that kind of head.

I'm working through how a linear map between consecutive position vectors could make L0H3's query-key score peak at distance one, which would need Wq and Wk to be related by something like a shift operator. That relation doesn't pin the map down, so I can't read the head's offset from the weights alone.

I'm checking whether the eigen-angles of B, which would be ω_k·Δ, pin down the offset for a previous-token head. The angles run from about 35° to 84°, mostly around 40–60°, so I'm testing what Δ=1 would imply for the frequencies.

I'm considering whether the eigen-angles, roughly 35° to 84°, actually fix the offset. If the position embeddings carry a range of frequencies, I'd expect the head to use only some of them, and the strongest components may be the high-frequency ones that give sharp previous-token attention. I'll print every eigenvalue above the noise floor, with its angle, to see whether any angles fall above 90°. Wrapping under a larger Δ could also change the reading.

I'm checking the eigen-angles of this head's bilinear form, which sit around 45° on average, and asking whether that fits a local-averaging head or a previous-token head. A pure self-attention head would sit near 0°, so this one is offset by some positive amount, and I'm leaning toward calling it previous-token.

I'm going through the layer-0 heads' value and key reads to see which ones carry token identity versus position. I'm checking L0H0's eigen-structure next, which looks like a mix of local and self-attending behavior, and then I want to estimate how many distinct tokens the model actually uses by looking at the effective rank of the token subspace.

I'm now considering whether the token subspace can be counted: I could take the union of the layer-0 readers, strip out the positional subspace that L0H3 and L0H5 read, and look at the effective rank that remains.

I'm now looking for induction-style matching: whether a later head's keys built from the previous token line up with the current token in the query. The English and Python data would show repeated names and keywords that such a head could exploit, while the romaji text would mostly be handled by local n-gram patterns, so the repetition alone won't tell me which corpus this is.

I'm checking whether layer-1 heads use the current token in their queries and the previous token in their keys, which would make an induction-style match: attending to positions whose previous token equals mine, then copying the token that followed. The other pairings, like matching on the previous token from the query side, look less natural, so I'm leaning toward the induction reading.

I'm testing whether the L1 head's bilinear form acts as an induction match: the current token in the query, the previous token at the key, and the OV map feeding back the token that followed. I'd check that by looking at the T-to-O3 block for a diagonal-dominant, identity-like structure.

I'm checking the eigenvalues of the restricted map P_T Q^T K OV3 P_T to see whether the symmetric part is positive and dominant, which would mean the head matches identical tokens, the signature of an induction circuit. I'll compare the reverse pairing too, using OV3^T Q^T K on the token subspace, and then account for the layer-one normalization.

I'm checking whether the layer-0 readers' input subspace is the same as the raw residual x-space, since the layer-norm weights only rescale per dimension and don't change which subspace the reads occupy. The argument holds: the normalization is a scalar per vector plus mean subtraction, so T and P remain valid subspaces of the raw residual stream.

I'm setting up the induction test for the layer-1 heads: projecting each head's query-key form and the OV3 map onto the token subspace T, then checking whether the eigenvalues are real and positive. I'm planning to repeat it for the later heads, and to use a larger T built from the value reads of L0H0 and L0H3.

