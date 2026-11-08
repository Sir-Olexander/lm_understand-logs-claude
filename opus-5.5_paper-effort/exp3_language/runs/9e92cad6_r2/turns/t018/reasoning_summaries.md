I'm working out the logit contribution of the L1 key-only vector: its norm is around 3-4 in normalized residual units, mostly along the outs direction. So its contribution to a key's logit is roughly 3.6 times how far that key's normalized residual extends along outs, and I'm checking whether that outs component is large enough to matter.

I'm working out that the L1 heads all seem to get a strong, query-independent key bias from the outs code, which would make every head attend to the current line or the most recent positions regardless of the query. That would be unusual, since I'd expect at least some content-based heads like induction.

I'm considering whether the outs-driven selection could instead be a recency signal, which would make the L1 heads local, attending to the most recent positions. I'm weighing whether the anti-duplicate term would push attention away from the same token, though that would break down when the previous token matches the current one.

I'm checking whether the outs code could be a plain running mean of position, which would only give a weak recency signal at large positions, so a newline-weighted content code would make more sense and point toward line-level attention.

I'm now asking whether the outs code might instead be position itself, not newline-ness: a running mean of position would give almost no recency signal at large positions, since its increments shrink as 1/j. That makes the L0H5 readout look like a weak, slowly varying code, and I'm reconsidering what the ins subspace actually encodes.

I'm checking whether the ins subspace could be token identity after all, since space and newline embeddings might carry components there too. But MLP0 reading ins at only 0.6× makes that unlikely, since it would need to know whether the current character is a space. I'm about to test whether any later attention value reads ins.

I'm checking whether the L0 group is really computing running means of position, with the weights set by the query's position and the key's token, so the logit for each pair depends on both.

Checking whether newlines are frequent enough in these corpora to make line-level attention worth it: English Wikipedia paragraphs run hundreds of characters, so newlines would be rare in 128-token windows, while romaji Aozora dialogue lines and Python code, with their short lines, would have them often.

I'm looking at whether the group's key-side vector leans on token identity, and it does: about 44% of its energy sits in the token subspace, with only a small share in the positional one. Since the four heads differ in their scaling, they may be computing weighted running means of newline positions over different time spans.

I'm trying to pin down what combination of the four L0 group heads' outputs the L1 key-only vector r picks out, since those heads share the two-dimensional outs subspace with slightly different output directions. I'm also wondering whether the model needs line-position information to predict newlines, which would mean it has to track how many characters are already in the current line.

I'm thinking about the newline token itself as a counter: uniform attention over the span from the last newline to the current position would give it weight 1/(L+1), so its value would encode the current line length. I want stronger evidence on whether the L1–L3 attention really is uniform within the current line, so I'm going to check how much the content-dependent QK part matters next to the key-only bias.

I'm estimating the size of the key-only term: with the residual norm around 19.6 and the outs share of its energy, the contribution to a logit is only a few units, so I need to see whether the L0 group's larger outputs dominate the layer-1 input.

I'm weighing whether the L0 group's outputs or the MLP0 output dominate the layer-1 residual, which would decide how much the key-only bias matters. I'm also considering a test: in shuffled text query content shouldn't matter to any head except through the current token, so later content-content QK should be weak if the corpus is shuffled, while in as-is text it should show structure.

I'm checking whether the L1 heads' strong negative block on same-token pairs is a structured content effect, like induction or a vowel/consonant pattern, or just noise, since "avoid same token" might mean excluding the query's own token from a line average.

I'm looking at whether L3H3 behaves like an induction head: its QK is nearly positive-definite, so it matches query and key features in the same subspace, and its OV copies the token signal from L0H3's output. That suggests it attends to keys whose residual features resemble the query's, while the L1 heads may be doing something different entirely.

I'm checking whether L0H3's output subspace lines up with the features that L3H3's query and key share, since that would confirm the induction-style matching I suspect.

I'm checking the L3H3 query-side reads against the L0H3 output subspace, where the query reads the token signal at 1.5× and the output subspace at 1.6×, and the key reads the token at 1.0× and the output subspace at 1.9×. Its QK is nearly symmetric positive-definite, so it matches similar features, but the direct token-to-token term is small, which points to later-layer features rather than raw tokens.

I'm asking whether L3H3 is matching on a same-line bag summary built by the L1 and L2 heads, which would make it attend to positions in the same line or lines with similar makeup. For as-is text I'm considering a looser version, matching on similar context, though attending to a same-context position wouldn't obviously predict the next token.

I'm asking whether the L3H3 pattern is really induction. Its value would have to carry the token at j+1 rather than the prediction made at j, and a clean positive-definite QK in the residual basis would need the shifted features to share the same subspace as the unshifted ones, which seems unlikely.

I'm weighing whether L3H3's query-key match is a mix of two effects: a same-bigram match from the L0H3-copied previous-token features on both sides, and a current-token-to-previous-token cross term that would look more like induction. I still need to gauge which of the two is stronger.

I'm breaking L3H3's QK into blocks over the query sources (the current token T and the L0H3 output subspace) and the matching key sources, normalizing by subspace size to see whether the positive-definite part comes from the same-token term or the cross term.

