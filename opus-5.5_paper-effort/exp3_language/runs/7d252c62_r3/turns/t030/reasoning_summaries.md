Two heads look like they're tracking the same offset structure. L2H1 and L3H5 both show a strong attention pattern from the token T to the offset-5 position, and from offset-5 to offset-1, each clearly stronger than the reverse direction. That asymmetry fits an induction-style alignment if offset-5 corresponds to offset 1 and offset-1 to offset 2. L3H5 also has large diagonal scores at both of those positions, which strengthens the reading.

Now I'm reading the L1 heads, which show the reverse asymmetry, and I'm working out what that could mean for matching the previous bigram to a shifted earlier position, so later L2 heads can pick up on it.

Now I'm checking the alternate convention, where o1 is offset 1 and o5 is offset 2, to see whether any head fits induction under it. Under that reading, L1 shows a strong pattern but L2H1 and L3H5 are weak, so no head is consistently inducting.

The asymmetry in L1 points toward a head that links the query's previous-token context to an earlier key position, which later L2 heads could build on. I'm weighing whether the offset-5 and offset-1 reading holds up on its own terms, since the two-pair consistency within L2H1 and L3H5 is what makes me lean toward it over the alternate convention.

I'm checking whether the L1 pattern could be a prev-token effect: when the key sits right before the query, its token trivially matches the query's previous token, which might explain the reversed asymmetry.

I'm noticing that the L1 heads' reversed asymmetry might just be a content-based previous-token mechanism, since matching the query's previous token against the key's current token would naturally pull attention to position i-1. I'm wondering whether six such heads would each copy different features from that prior position.

Now I'm reading the L0 heads, which seem to handle the immediately preceding position (offset 1) and the one before it (offset 2). The L1 heads may be pulling in recent positions through content matching. Several heads look like induction or copy heads, with L3H5 showing the strongest long-range alignment.

Still checking the offset pattern in L3H5: the diagonal scores suggest it also matches the position analogous to mine, so the match looks fuzzy rather than strictly one step back. Overlapping offset coverage in L0H5 and L0H1 might explain that.

Now I'm thinking about what evidence would separate these hypotheses. Python code is highly repetitive and depends on indentation copied from the previous line, which could call for a head that attends to the start of the prior line. Romaji text, with its limited syllable inventory and frequent particles, offers another test case for repeated-token copying.

Now I'm weighing whether the token subspace rank could reveal the corpus, since unused romaji letters like l, q, v, and x would leave their embeddings untrained and shrink the effective dimension to around 23, compared with roughly 27 for other text. The eigenvalue drop near 21 to 25 is what I'd need to check next.

I'm looking at the singular values of L0H5's OV map, which decay gradually with a possible break somewhere around the 19th to 21st component. That break could mark where the signal ends and the tail of smaller values begins, so I'm checking whether the tail sits above the noise floor.

I'm wondering whether the spectrum's tail is just random-init noise or carries something real, like positional information or rare tokens. Token frequency could also explain the uneven embedding norms, though the OV spectrum isn't a clean count of tokens. Newline-focused heads are another possibility, since indentation copying would require attending to each previous line's start, but I'd first need to identify which token is the newline.

Now I'm considering whether the newline token has its own direction in the embedding space. Heads H2 and H3 in L0 seem to key on a specific token direction when the query is positional, which could mark a "most recent newline" head, and H1 and H5 share a key-token direction that might be a barrier token like newline or space. I can't tell how much of the embedding is spent on newlines versus letters. I'm also thinking romaji's consonant-vowel structure would give the token geometry a dominant vowel-versus-consonant axis, stronger than in English, while Python would look different.

I'm weighing whether the number of distinct romaji letters, around 24 counting space and newline, would show up as a reduced effective rank in the token subspace.

Now I'm working out how to count the tokens from the weights: the OV eigenvalues look gradual, so the drop near 24 is not clear yet. I'm going to look at an early MLP input restricted to the non-positional subspace, which should handle every token directly.

I'm weighing whether the number of distinct tokens in the corpus could show up in the layer-0 embedding span, since positions take up some of those dimensions too and I can't cleanly separate the two. Python would be different from romaji here: its indentation and newlines would push some heads toward attending to line starts, which might be a clue to check.

I'm considering whether the prev-token mechanism in layer 1 could show up in the weights, since matching the query's previous token against the key's current token would pull attention toward the prior position. I'm also wondering whether the vowel-consonant alternation in romaji would leave a visible trace there.

I'm checking whether the "same token" matching in that bilinear form is positive, which would confirm a content-based previous-token mechanism. I'm wondering whether I can recover the actual token identities from the token subspace, perhaps by looking for neurons that respond to specific tokens.

I'm weighing whether indentation-copying in Python would show up as heads tied to space runs at line starts, since the model has to count the leading spaces and carry them over from the previous line.

I'm considering how indentation runs would be handled: a head inside the indentation might attend to the matching position in the previous line, which is an induction-style pattern on positions. English text has long lines with few newlines in the window, so newline prediction there would hinge on sentence endings instead. I'm also wondering whether the position-in-line signal matters more than absolute position for code, though that question is still open.

I'm trying to get a handle on the number of distinct tokens by looking at the token-only part of layer 0. I'll check OV5's singular values against a random baseline to see where the real signal ends.

I'm looking at the value matrix of the offset-5 head, and its singular values fall off gradually with no clean break, so I'm not seeing a sharp boundary that would reveal the token count. I'm considering a different route: the token embeddings should form about 28 points in a subspace, and checking the output of the offset-5 head through later layers might expose how many token dimensions are actually in use.

I'm checking whether the combined token-reading directions across OV1, OV5, and the layer-0 MLP input span the full token space, which would give a count of distinct tokens. I'm also noting that layer norm centering removes one dimension, so the expected rank may be off by one from the number of tokens.

Now I'm thinking about whether the full layer-0 input weights could reveal a rank gap near the combined token and position dimensions, since every learned input direction has to lie inside the span of the normalized embeddings.

I'm checking the layer-0 MLP input weights, where the token subspace should dominate because the layer-norm gain on those inputs is large. I want to see whether the top singular directions of that matrix are mostly token directions.

I'm looking at whether word-length statistics could separate the corpora, since English words average around five letters with a space while romaji syllables run shorter. Python's indentation runs would show up as long space sequences at line starts, a different signature from either.

I'm weighing whether newline frequency could separate Python from English prose: Python has a newline roughly every 35 characters, while Wikipedia paragraphs run long. Word-level cues like the space-delimited romaji particles might give a similar signal, but I can't count tokens directly from these weights yet.

Now I'm weighing whether the bilinear prev-token matching in layer 1 could show up as a token-token matrix in the token coordinates, which would be positive definite if the head matches identical tokens.

I'm asking whether the bilinear form in layer 1 could recover individual token embeddings, if the head matches identical tokens. If it does, the token-token matrix should look like a Gram-type structure, with its largest entries on the diagonal, and that might reveal how many distinct tokens there are. I'm not sure it's invertible to the individual embeddings without knowing them already.

I'm wondering whether the bilinear form could expose individual token embeddings if the head matches identical tokens, though it probably can't be inverted without already knowing them. I'm also considering whether the residual-stream geometry of tokens could be probed through the positional heads' key preferences, and which corpus would make layer 0 average over positions so heavily.

I'm thinking about whether the token-direction question might be settled by frequency: since space is the most common token, its direction would dominate, so I'd check whether the strongest key direction in L0 behaves like a space or a newline.

I'm looking at whether the L1 content-match head's bilinear form, restricted to token coordinates, could show a sharp rank drop at the number of distinct tokens. I'd compute its singular values for that, though I'm not certain the query path through layer norm leaves a clean structure to read off.

I'm going to build a larger token subspace estimate from the top singular directions of OV5 and OV1, minus the positional part. Then I'll compute the L1 bilinear form on that subspace for each head and check where its singular values drop, to get an effective rank near the number of tokens.

