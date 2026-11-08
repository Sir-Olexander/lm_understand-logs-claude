I'm checking which heads read the 53° plane: L0H2's output gain is about average and MLP0 reads it at about average, while L0H2's queries and keys respond strongly. The S-heads' values and queries are roughly eight to ten times more enriched for it than their averages, and L0H3's queries and keys follow the same pattern.

I'm weighing whether the S-heads' query could track absolute position in this plane, since the rotation of 52.6° per step would make attention shift with position mod 6.84. That seems odd for a position-agnostic task, but the oscillating query dot product might just be a side effect of how the weights combine.

I'm weighing whether the corpus could be Python or English Wikipedia, since Python would show a strong space-space bigram from indentation and frequent newlines. Now I'm turning to the induction head in L3H1, whose query-key score is moderate, to check whether an earlier L2 head is doing that job with a better measure.

I'm checking whether L1 heads could take over the previous-token role, since that would give L2 induction keys built from trigram-like features. Measuring induction scores for heads in layers 1 through 3 with a cleaner token subspace, I find only L3H1 scores strongly, at 3.15, while everything else stays at or below 0.75.

I'm noting that L3H0 and L2H1 both score moderately on the reversed query-key pattern, where the query reads the previous token and the key reads the current one. That would make them attend to an earlier occurrence of the character before them, which doesn't directly give the next token, so it's probably not a standard induction variant.

I'm checking whether the induction behavior is concentrated in one head: L3H1 scores strongly on the cleaner test, while the others stay low, with L3H0 and L2H1 showing only moderate scores on a reversed pattern that doesn't map cleanly onto standard induction.

I'm checking L3H5 next: its query-key norm is small, so it might attend broadly, and its value reads mostly from the residual rest rather than the current-token subspace. I'm also wondering whether L0H2's input is limited to the dimensions that layer-0 normalization keeps large, since that would change which token features it can actually see.

I'm working through why the token-subspace measure might be incomplete: the embedding components in dimensions with low layer-0 norm weights never reach L0H2's input, so later heads reading token identity elsewhere would look weakly aligned. That makes the current overlap numbers an underestimate. Turning to the corpus, I'm considering Python text, where stripping punctuation and digits would leave things like "self x foo a b" and underscored identifiers split into words.

I'm looking at the output of the final layer and which heads write into its top direction, where the L3 heads show enrichment of about three to four times and L0H0 is at about 2.8. I'm also weighing whether a few later dimensions with large values point to a separate newline-related mechanism, though I can't tie them to specific tokens yet.

I'm looking at which final-layer heads write into the top output directions, and the L3 heads show enrichment of about three to four times there. I'm also weighing whether combining the residual writers across the last layers could reveal a logit-relevant subspace.

I'm looking at what the final layer's output can actually respond to: the final norm and unembedding only read about 28 directions, so gradient pressure on the last projection stays inside that span, plus the direction of the normalized residual itself.

I'm working out why the final layer's gradient should stay in a space of about 28 dimensions: the unembedding only reads that many directions, and the leakage through the layer norm adds a few more. I'm checking whether the extra singular values fit that, and whether the unused tokens in romaji would contribute almost nothing to the gradient.

I'm wondering whether the medium singular values could be leftover imprints from early training, when the unused romaji-token directions still got consistent gradient pushes before weight decay faded them. That would make the roughly five medium directions a sign of unused tokens rather than a separate mechanism.

I'm weighing whether the rare letters might still carry strong logit directions in English, since they'd appear about a million times each over a billion characters. I'm also checking how Python identifiers like sqrt, query, and queue would load onto q, z, and j in a comparable way.

I'm working out whether the top final-layer directions reflect a logit-relevant subspace, comparing W_proj.3's leading directions against W_o.3's and the earlier layers'. I'm also considering a different angle: the learned positional structure may reflect the relative-position information the model needs, with the slower planes possibly carrying Python-style positional patterns.

I'm checking whether the negative same-token attention in L1 could mean the heads gather information from other characters, since double letters and sokuon doubles like "kk" and "tt" are common in romaji.

I'm weighing whether the negative same-token attention in L1 could mean those heads gather letters that differ from the current one, perhaps building a picture of which characters are already present in the line. I'm also checking whether L0H2 might attend locally to the newline rather than to a fixed previous position, since that would change how I read its sharpness.

I'm settling on L0H2 as a purely positional head: its query-key match is almost entirely in the positional subspace, so it attends to a fixed offset whatever the content is. That makes it a poor fit for shuffled text, where only line boundaries carry meaning. I'd put about 90% confidence on that reading, though a previous-token copy could still help when the preceding character is a newline.

I'm turning to the corpus question now: whether the romaji-style CV alternation would show up in the weights, and whether previous-two-character context would be what matters most for the L0H2 head.

I'm weighing whether the romaji-style alternation of consonants and vowels, plus the spacing a romanizer like pykakasi adds between words, would show up in the weights. For Python the newline would be far more frequent and structurally important, so heads that track the position within a line would matter much more there than in English prose.

I'm considering that the S-feature only carries real signal if attention concentrates on one or two recent occurrences of the key character, since averaging phasors over many spaces would wash out toward zero. That would make it meaningful mainly in text where the chosen character is rare, like newlines in Python.

I'm testing whether the 53° plane could be a slowly rotating positional component, so that L0H3's scores would oscillate with offset and favor positions a fixed period apart. That pattern seems odd for any corpus, which makes me doubt the plane is positional at all and suggests it might encode something else, like a large per-position signal.

I'm considering whether L0H3's query-key match in that plane is a plain positive-definite form, so its scores would favor closer positions rather than oscillate with offset. I can't settle whether the plane tracks a monotone position signal without the actual positional embeddings.

I'm checking whether the 53° plane encodes position on a slow curve rather than a per-step rotation. If phase changes slowly across positions, a positive-definite score would favor nearby positions, which makes L0H3 look like a local head. But a 52.6° per-step rotation would make L0H2 reach a large offset, and I'm not sure that fits its sharp, near-previous attention.

I'm wondering whether the 53° plane might carry a slowly varying absolute-position signal that L0H2 reads with a large offset, while the faster planes supply the sharpness. I can't pin down the full mechanism without the actual positional embeddings, so I'm treating the reconstruction as too speculative to push further.

I'm considering that the averaged position of earlier content-selected tokens might step up each time a new one appears, so later segments would carry higher values, and L1 keys could favor the current segment through that recency signal.

The 53° plane looks like a slowly varying absolute-position signal, which would let the S-heads' queries scale their attention to the key character as the context grows. L0H3 would then act as a local head, favoring nearby positions through its positive-definite form on that plane, though I can't confirm the exact mechanism without the positional embeddings.

I'm working out whether the key-selected token X is a space or a newline, since the S-heads' query scaling depends on which one it is. The 53° plane's slow positional signal would make the S-heads' attention to X's scale with context length, so the distinction matters for the mechanism.

I'm comparing the S-heads' key vectors to see whether they select different characters. The overlaps between H0, H1, H4 and H5 are uneven, so each head may be attending to a different key character.

I'm reconsidering whether the S-heads' key character is a space or a newline, since that determines whether they segment by word or by line. The four S-heads select different key tokens (their key overlaps are uneven), so they likely each track a different character type while writing into the same channel.

I'm considering whether the MLP0 neurons' input weights could separate out individual token directions, but with 384-dimensional inputs and only about 27 token dimensions to pick apart, that doesn't look separable. I'll switch to general cues instead: a Python corpus would show strong newline and indentation structure with heavy repetition, and we only see one induction head at L3, so Python seems less likely than English.

I'm now looking at the composition of the L2 keys, asking whether the bulk of their signal comes from L1 outputs. I plan to measure that by projecting onto the L1 OV output subspaces and taking the top components, then doing the same for L3 keys against L2 outputs. I also want to check how much the keys draw from the MLP0 output, which would round out the Q/K/V breakdown for each head.

I'm now planning to measure how much each head's queries, keys, and values draw from earlier sources, such as the attention and MLP outputs of earlier layers. I'd score each source by how strongly its output subspace aligns with the head's read direction, normalized against random alignment, and I'd cover the MLP outputs too.

