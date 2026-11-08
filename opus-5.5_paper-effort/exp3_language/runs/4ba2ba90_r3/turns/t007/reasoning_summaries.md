I'm noting that L0H2's query-key matching runs on the positional embeddings and its output reads the token signal, which holds by construction. The broad heads L0H0, H1, H4 and H5 seem to read positional information on their value inputs and their query side too, which looks odd for heads that should be averaging uniformly.

I'm now thinking the broad heads might be sharp content-based heads after all: L0H0 could key on space or newline tokens and copy their positions, which would explain why a near-zero QK norm still gives strong attention. So my earlier "uniform averaging" reading looks premature.

Now I'm checking whether the L1 heads' keys work the same way: their keys read both position and token content, and the query-independent key preference from the query bias may matter more than the query side. I want to compare that bias-driven term against the query-dependent part.

Now I'm checking the L1 heads' keys, which read both position and token content, and I'm seeing that L3H1 is an induction head: it matches the current token against the previous token that L0H2 copies, with a strong positive score of +0.94. That's the classic pattern, and it fits the earlier K-composition result with L0H2 at +0.84. The L1 heads, by contrast, show only weak induction scores around +0.33.

I'm now looking at a few candidate features that could tell Python, romaji, and English apart. Indentation stands out for Python: lines start with runs of spaces, and predicting how many means copying the previous line's indent, which might be done by a head that tracks the last newline. That fits the broad L0 heads that key on token content and copy position.

I'm now weighing a few ways to tell Python, romaji, and English apart. Word-start position, meaning distance since the last space, would help with English and romaji word lengths, so a "last space" head would be useful across all three. The remaining question is whether the token embeddings span a low-dimensional subspace, since unused tokens would receive no gradient and stay near their initial values.

I'm working out how many dimensions the token subspace really spans: with unused tokens getting no gradient, the learned span should be roughly the number of tokens actually in use, possibly minus one from the layer-norm centering. Adding the positional part and projecting it out to isolate the token signal, I'm now counting the significant singular values in L0H2's OV spectrum, which so far read 0.79, 0.72, and 0.68 before a zero.

I'm sorting through the L0 output-side singular values to see where the meaningful directions stop. The spectrum falls off gradually with no clear gap, so counting dimensions that way doesn't settle it. I'm wondering whether the five weaker directions correspond to rarer tokens that still show up in the data.

I'm now listing the romaji letters that would show up in the data: most of the alphabet, plus a few rare ones like v, and l, q, x only from embedded Latin words. I'm also considering how small-tsu doubling in Hepburn output would make repeated consonants like "tt" or "kk" common.

I'm now going through the romaji letters: the Hepburn output uses about 22 letters, plus space and newline, which gives roughly 24 common tokens and about 23 significant dimensions. That lines up with the 23 strong singular values, which would leave the four rarer letters (l, q, v, x) to account for the weaker tail.

Now I'm weighing whether English and romaji would both produce the same 23-strong-dimension structure, since each has roughly 24 common tokens plus a few rare letters. Python looks different, with letter frequencies in code shifting toward e, t, s, a and away from the rarer letters, so I'm starting to check what its spectrum would look like.

I'm weighing whether Python, romaji, and English would each leave the same signature in the spectrum, since all three seem to have around 24 common tokens plus a few rare letters. Python's rare letters (q, z, j, and to a lesser degree k, v, w, x, y) differ in which characters they are, so the rare tail might carry different signal even if its size doesn't separate the three.

I'm weighing the weak output directions against what romaji would actually produce: if l, q, x and v never appear, their logits would be flat and leave no direction. Since the weaker singular values sit around 70-80% of the 23rd, they look like rare but present tokens, which favors English over romaji.

I'm considering whether the rare romaji letters like v might still produce moderate directions, which would make the spectrum hard to read. Direction 25 is unusually sparse, which suggests a few dedicated neurons, maybe tracking a rare token or a pattern like newlines or indentation.

I'm now thinking about how Python's indentation and line structure would make newline and space positions matter, since lines start with runs of leading spaces, which the other corpora don't share in the same way.

I'm now picturing how Python's indentation would need a head that compares a line's leading spaces with the previous line's, attending to the same column in the prior line. Python's repeated identifiers also suggest induction heads matter a lot, and L3H1 is already a clear one, while English and romaji text repeat names and phrases too.

I'm looking at whether the spaces that separate identifiers in normalized Python, where symbols like parentheses and dots become spaces, could be tracked through the weights. I wonder if the L0 heads' token-keyed attention reveals which tokens they respond to, though without the embeddings I can't tell which token that is. The L3H1 induction head's OV circuit might be a way in, or perhaps the token embeddings could be reconstructed from the weights themselves.

I'm trying to tell whether the token embeddings are nearly orthogonal, since that would let the bilinear form M from L3H1's query and key projections act like a near-identity match on each token. If so, the form should be large for same-token pairs and small otherwise, so it could expose which tokens are in play without the embeddings themselves.

Next I'm considering whether the token embeddings are close to orthogonal, so the eigenvectors of the bilinear form might line up with individual tokens if its weights are uneven. I'm also thinking the MLP0 neurons could cluster around particular token directions, which might reveal which tokens are present. Another angle is the regular consonant-vowel alternation in romaji, versus the identifier-heavy structure of Python.

I'm now checking whether the L1 heads are all local heads at different offsets, since they share a strong positional key read and similar bias terms, and I'll look at their query-key structure restricted to the positional subspace.

I'm checking whether L1's positional blocks are just L0H2's rotations at a different offset: if L0H2 rotates by ω·1 in each plane, a head at offset 2 should rotate by twice that angle in the same planes, so I'll compare phases in L0H2's eigenbasis.

I'm checking whether the positional rotation in L0H2 carries over to the other heads: if each plane rotates at a fixed frequency, a head at a different offset should show the same planes with angles scaled by the offset ratio. I'll take L0H2's complex eigenplanes as the basis and measure each other head's 2x2 block in those planes.

I'm defining the rotation angle and reflection magnitude for each head's 2x2 block in L0H2's eigenplanes, and I'm expecting a head at offset d' to show angles scaled by d'/d. I'll apply that to all the heads and check the broad ones separately, since they may not fit the pattern.

