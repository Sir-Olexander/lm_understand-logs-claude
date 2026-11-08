Later heads' rotations on these planes are much smaller than L0H2's, with only tiny components showing up there. That probably means the raw positional signal has less influence on their query-key matching, possibly because layer normalization shrinks the positional part as the residual stream grows, so the values are small but still meaningful in relative terms. Now I'm turning to L3H1, where the planes I'm checking all seem to sit near zero angle.

I'm checking whether L3H1's planes at near-zero angle mean it's matching positions that are close together, which would look like a local or recency bias. That reading fits with L3H0 also sitting near zero. I'm also noting that the L1 plane angles are hard to interpret, and I can't pin down the offset without knowing the frequencies.

Now I'm checking whether L3H1's induction-style matching works with L0H2 attending one token back, since that would explain how it copies the token that follows an earlier occurrence. The L1 heads look similar to each other, with strong negative weight on same-token pairs and most of their key signal coming from position, so I'm wondering whether they're broad positional heads.

I'm looking at whether L1's heads spread attention broadly, since their key signal comes mostly from position and their OV outputs seem to push against the input. I'm also thinking about whether their query bias gives the keys a fixed preference, which could explain why they look sharp given their large norms.

I'm trying to compare the size of g against the typical query-dependent term, and to project it onto the positional and token subspaces. But without knowing the direction of position increase, the positional analysis stalls, so I'll shift to deciding head types from composition and QK structure instead. L0H2 stands out as the previous-token head, with its strong positional rotation in QK and high-rank OV reading the token subspace. Heads L0H0, H1, H4 and H5 have much smaller QK norms, which makes me wonder whether their attention is close to uniform.

I'm estimating how selective these heads could be: with a rank-one QK of σ1 ≈ 0.41 and typical projections around 10, the scores span roughly 5, which is moderate. Their OV reads mostly positional content, so they seem to copy positional information from whatever they attend to. If attention were close to uniform, that would amount to a smooth average of position embeddings, which makes me wonder whether these heads are doing something more specific, like attending to particular tokens such as spaces or newlines.

I'm checking whether the keys for these heads read token identity strongly enough to pick out particular tokens, like spaces, with the query's positional signal scaling how strongly they attend. Since their outputs mostly carry positional information, they may be recording where the last such token sat, which would give a sense of word length so far.

I'm grouping L0H0, H1, H4 and H5 as broad heads, since their weak QK structure suggests spread-out attention, with OV outputs that copy positional information from many earlier tokens.

Still checking whether L0's broad heads are just averaging positions, which would make them redundant, so I'm considering whether the token-dependent part of their keys changes that picture.

Now I'm weighing whether L0H2's QK structure is really positional or whether it reads token identity, since the Q and K subspaces nearly coincide and the spectrum is fairly flat. If the broad heads' OV circuits read the other subspace, they would be averaging token content rather than position, which would look like a bag-of-characters signal.

I'm counting the dominant eigenvalue pairs in L0H2's QK spectrum, which gives about 20 dimensions from the complex pairs plus three real ones, so roughly 23 significant directions in total.

I'm checking whether L0H2's output carries positional information about the source token, which would mean the induction-style match in L3H1 depends on whether the matched position lines up with the query position.

I'm working out whether L0H2's rotation-like structure reads as positional or token content, and I'm weighing that against the broad heads' OV circuits, since a positional reading would fit an absolute-position mechanism while a content reading would fit a token-level one.

I'm weighing whether L0H2's QK structure is really about token content instead of position, which would flip the roles: the broad heads would then be averaging token identities rather than positions. I'm leaning toward keeping the original reading, where Ppos is positional and L0H2 is a previous-token head, while still keeping the bipartite content explanation open.

I'm now weighing the reverse hypothesis, where the positional subspace is actually the token subspace, so the broad heads would be averaging token identities. I'm asking why a token-selective head would copy position, and whether a rank-23 token-token interaction could really be doing "attend to the last newline."

I'm weighing the dimensionality of each subspace as a clue: the token subspace can hold at most about 28 dimensions, while a learned positional subspace can be larger, and L0H2's QK spectrum runs to roughly 23 dimensions with the OV input spread over a similar range. Next I want to see what the unembedding says about which subspace carries token content.

I'm weighing whether the token subspace or the positional one is which, since the broad heads' outputs and L0's MLP could read either. The helix-like rotation structure is the clearest sign of positional embeddings, while token-token interactions wouldn't organize into neat conjugate pairs like this.

I'm weighing the positivity of the L3H1 match against L0H2's output: if the attended position is always at or before j, then matching the query's position would favor keys whose attended source sits near the current position.

I'm weighing the induction explanation for L3H1 against the alternative, and the usual reading still looks more likely. Under that reading, the same-token suppression in the L1 heads fits with avoiding repeats, while the other reading would make it a broad positional effect. The stronger test is L3H1's output: it reads tokens heavily, not positions, which is what a copying head should do.

I'm weighing the unembedding test: if the composed path from the token embedding through L3H1's OV should map each token back to its own logit, that would confirm the copying reading, though I can't run that check without the embedding and output matrices.

I'm now checking how big the positional subspace really is: I'd gather the union of the input directions that L0's query, key, value and MLP weights read, and see whether the token part fits within roughly 28 dimensions.

I'm checking whether the positional subspace is the larger one by comparing its dimension count against the roughly 28-dimension token budget, starting from where L0H2's QK acts and where its OV input spreads.

I'm weighing the positional and token subspaces by how each enters the query and key sides, and I'm checking whether the layer-norm bias on the first layer's input changes which direction looks like content. I'm also considering whether an attention-sink or BOS token would give a clue, though nothing in this setup points to one. Under the induction reading the query reads the current token and the keys read a prior-token representation, which should land in a separate output subspace from both the token and positional directions, so I'm leaning toward that explanation.

I'm checking whether L3H1's induction-style match holds up across the later heads, and whether swapping the token subspace for the positional one gives a meaningful score. I also want to know how much of L3H1's query reads the token subspace versus the positional one. The same-token negativity across all the L1 heads is the part I can't yet explain, since it would look odd under the positional reading.

I'm trying to see whether the broad heads' OV inputs could span the whole token space, given how low their OV ranks are. I'm now testing the dimensionality directly: a token subspace should fit within about 27 dimensions, so I'll take the union of L0's input directions and check whether a ~27-dimensional slice can carry the token content while the rest holds the positional part.

Low-rank OV in the broad heads points to them copying something low-dimensional like position, which fits H_A. L0H2's high-rank OV looks more like token identity, roughly 25 dimensions against the 27-dimensional token space.

I'm checking the broad heads now: L0H0's query leans mostly on the positional part, and its key leans on both token and position, so I want to know how sharp that attention is. Before judging that, I need a rough sense of how large the token and positional components are after layer norm.

I'm now looking at L0H3, whose QK has a top singular value near 2.87 and roughly rank two, with both its query and key sides leaning heavily on the positional subspace.

I'm turning to the first layer's heads with a same-token suppression pattern, where the keys read both positional and token directions. I'll check how strongly the keys favor each of those, and whether the query side adds any bias.

I'm checking whether the L1 heads' keys are driven by the broad heads' outputs, which carry positional information. Their raw positional planes are small next to the large QK norms, so the matching probably runs through the L0 composition instead.

I'm now sorting the L1 heads by which inputs drive their matching, whether the raw residual or the broad heads' outputs, to see whether they're tracking the same word or the same line.

I'm now computing, for each of the later heads, how much of its query and key sides comes from each source: the raw token and positional directions, each layer-0 head's output, and the layer-0 MLP output. I'll use projection norms since these subspaces overlap, so each source's share is only approximate.

I'm now going to work out how much of each later head's query and key comes from each source, looking at the token and positional directions, each layer-0 head's output, and the layer-0 MLP output. Since those subspaces overlap, I'll treat the projection norms as only a rough breakdown. I'm not sure the magnitudes will tell me much, so I may instead look at which sources push matching in the same direction.

I'm weighing how the output-direction count might track token frequency: rarer tokens should get less gradient, so their learned directions would be weaker, which could explain the five weaker dimensions alongside the 23 strong ones.

I'm working out whether the output-direction count fits a corpus: the rare English letters like q, x, and z should still get moderate direction strength from context, while romaji's rarer letters would give more mixed signals.

Now I'm comparing the output-direction strengths against what romaji would produce, since its rare letters should leave a few weak dimensions and its absent ones should leave almost none. The tail values near 3 and 4 could be weak signal rather than bulk noise.

The bulk edge should sit near 1.3, but the 40th value is 2.18 and the 80th is 1.43, so many dimensions rise above the noise floor. That means the learned structure isn't confined to about 27 dimensions, and there may be extra signal from something like residual leakage.

I'm treating the output-direction count as too fuzzy to settle the corpus question by itself, so I'm looking at a different kind of signal instead. Python's indentation runs in multiples of four spaces, so a mechanism that counts spaces from the previous newline or matches the prior line's indent would leave a distinct trace, though without positional embeddings that is hard to confirm.

Now I'm weighing the romaji idea against the rest: vowel-consonant alternation makes the next character highly predictable from the previous one, and that's a local pattern that English shares too. Repeated identifiers and "self" in Python would show up as induction behavior, which is strongest in L3H1 but only weakly present earlier.

I'm now thinking about whether the spectrum of the L0H2 output-value matrix could reveal how many distinct previous-token identities the model tracks, since a clean drop after about 24 singular values would point to a 24-letter romaji alphabet. The first few singular values I'm reading off are fairly flat, so I'll need the rest of the sequence before judging that.

I'm checking whether the decay in L0H2's output-value singular values, which falls off gradually rather than dropping cleanly, could tell me how many distinct previous-token identities the model tracks, and I'm looking at the value side too for the same pattern.

I'm considering whether the spectrum of the token-space eigenvalues could give a count of distinct tokens, since the absent letters in romaji would shrink the span of the embeddings that layer 0 reads. The L0H2 output-value spectrum decays too gradually to settle that by itself, so I'd need a cleaner measure of how many token directions actually participate.

I'm checking whether the L0H2 output-value spectrum has a clean elbow near 24 or 28 singular values, which would point to a token count, and comparing it against the random-init noise level of about 0.13 for V to see how many learned dimensions really remain.

I'm checking whether the L0H2 output-value spectrum has a clean drop that would match a 24-letter alphabet, since the bulk of the other ideas hinge on that count. I'm also considering an unembedding-based check: if the absent letters are missing from the output side, their rows should look unusually flat.

I'm weighing English against romaji by how often each letter shows up: the rare letters q, x, and z should get weak gradients, while common ones like e, t, and a should dominate the output directions.

I'm checking whether English's letter frequencies would produce the same split: about 23 common symbols with a few rare ones (x, j, q, z, newline) falling into a weaker tier, which would match the observed 23 strong and 5 weak output directions.

I'm counting the dimensions again: if the 28 tokens leave 27 difference dimensions, I'd expect about 22 strong and 5 weak, but the observed 23 strong plus 5 weaker plus one more doesn't fit that exactly. I'll stop trying to pin the count so precisely, since the Jacobian of the layer norm could add a mean direction. Next I'm sketching Python's letter frequencies, with e leading and the rest of the common letters trailing off.

I'm estimating how common each letter is in romaji text, with a, i, o, and u leading and the rarer consonants trailing off, so I can compare that spread against the Python and English patterns I was weighing.

Now I'm weighing English against Python on the rare letters: English has x, j, q, z and newline as weak tokens, and that would account for the five weaker dimensions I'm seeing.

I'm comparing the singular-value counts against English and Python letter frequencies: English would leave about five weak directions for rare letters like x, j, q, z and newline, which fits, while Python has fewer rare symbols and newline is common there. The one stronger direction near 8 with high kurtosis looks like a token tied to a few specific neurons, and I'm not yet sure which token that is.

I'm weighing English against Python on the newline and rare letters: if this is English, paragraph breaks would be the rare, specific token that makes the fifth weak direction, but Python's frequent newlines and indentation would push that count the other way. I'm also expecting Python to need strong induction heads earlier in the stack, and this model only shows a clear one in layer 3.

I'm checking whether L3H1's induction behavior holds up under better normalization, and whether other heads like L1H2 and L1H3 also feed its key through multi-token context matching. I'll also look at what the L1 heads are doing on their own, since they seem to be building simple previous-token or local context features.

I'm breaking the L1 query and key contributions into source subspaces: the token embedding, the positional part, the outputs of the broad L0 heads, the L0H2 output, and the MLP0 output. Then I'll score each query-source and key-source pair to see which ones drive attention.

I'm breaking the L1 query and key contributions into source subspaces and scoring each query-source and key-source pair to see which ones drive the attention.

