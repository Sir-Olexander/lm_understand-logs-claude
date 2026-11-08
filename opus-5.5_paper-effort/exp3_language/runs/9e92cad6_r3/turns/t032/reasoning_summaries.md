I've got an L3H3 induction score of 12.4, which is very strong, so it looks like a copy head that reads from L0H3's previous-token output. Still, the 0.41 baseline on random input subspaces means A@OV3 is biased toward positive values, so I need to account for that before trusting the result.

Now I'm checking whether the L3H3 positivity holds up under a longer match, since the query's previous token should line up with the key's prev-prev token if L0H3 covers offsets 1 and 2. I'm also testing whether a shuffled-text explanation could produce a score this strong, because lines that contain the current character would only weakly co-occur with the copied token.

Now I'm looking at the corpus for more evidence. The induction head only fires at the last layer here, and the earlier layers show only mild z-scores. I'm weighing whether a Python corpus with heavy repetition would show a stronger, earlier induction signal, versus comparing English against romaji. The copy OV's effective rank may help: if the head writes about 25 dimensions, that fits English letters plus space better than romaji, which has fewer active tokens.

I'm checking whether L3H3's OV on the raw token subspace holds up: restricted to that subspace and projected onto the top 28 directions of W_proj.3, its strength falls off gradually, with only about 20 to 21 dimensions above 0.7. The catch is that this subspace comes from L0H3's OV input and may not cover every token. A cleaner way to count active tokens might be to look at the input subspace directly, since tokens that never appear in training, like the romaji letters l, q, v, and x, get no learned response in layer 0.

Next I'm checking how L0H3's OV spectrum looks: about 18 dimensions carry strong weight, then it tapers smoothly. That suggests the head copies a limited set of token directions, and maybe groups tokens together. I'm thinking about approaching the count from the layer-0 MLP side, which reads the current token and the previous token to build bigram features.

Now I'm thinking about whether the MLP3 output, which carries most of its energy in about 28 directions, is simply writing into the row space of the unembedding. Tokens like l, q, v, and x that never show up in training would have their output rows pushed only by softmax gradients, so they might not add independent dimensions to that space.

I'm working out how the MLP3 energy is spread across dimensions: the two pairs at 40.6 and 34.1, and at 21.4 and 21.0, don't obviously fit the romaji count of 24 to 25, so they may come from something like LayerNorm scale directions. The rare letters are the open question for English, which would put the count nearer 28.

Now I'm checking whether the copy head's output lines up with MLP3's dominant directions: restricted to the top 24, its strength tapers without a clear cutoff, but at 28 dimensions it shows about 25 strong components, which points toward an English alphabet plus space rather than romaji.

I'm weighing whether English versus romaji can be told apart by a sharper count of the output dimensions, and the copy head's writes into MLP3's weaker directions look mixed rather than cleanly excluded. I'll try a different route: checking vowel-consonant alternation patterns instead of the dimension counts.

I'm weighing whether vowel versus consonant alternation in romaji could be encoded in one of MLP3's top output directions, and whether a five-dimensional vowel subspace would show that.

Now I'm checking the spectrum of L0H3's OV: eighteen dimensions carry strong weight, then it tapers in a smooth tail, and I'm trying to see whether 18 plus that tail can add up to a token count of 28.

I'm trying to see whether the tail of L0H3's OV spectrum, which keeps going down to about 0.06, reflects positional leakage or noise rather than real structure. The tail gets as high as about 0.2 at index 33, which seems to suggest more than 28 dimensions are doing something, so I'm going to try removing the positional subspace before trying to count anything.

I'm considering whether a Python corpus would show a stronger induction signal, since repeated identifiers like "self" and "x" give the copy head plenty to work with.

I'm weighing whether the "Romaji literature" and Python corpora would show a comparable induction signal, and I'm leaning toward giving Python a lower weight, around 0.2. On the English-versus-romaji question, the logit-dimension evidence favors English moderately, though the vowel-consonant alternation in romaji could mean the previous-token head matters less there.

Now I'm weighing whether a Japanese romaji corpus would show a stronger induction signal, since its long repeated morpheme sequences make copying useful, though those may be learned as n-grams anyway. I'm leaning toward English, and I want to recheck the output directions for the rare letters before settling on that.

I'm checking whether the large output-direction energies line up with English letter frequencies, since the pairs at 40.6 and 34.1 match the rare letters j and x, and q and z. The three medium-strength directions could correspond to v, k, and a newline or y, b, or w, with the space direction at the top. This is only a heuristic, since singular values don't guarantee one direction per token, so I'll look for a more direct test.

I'm considering whether a broad L3H4 head that boosts rare letters whenever they appear in context could explain the pattern, since foreign names in English text and foreign passages in romaji would both cluster those letters. That would make such a head useful in either setting, so it doesn't settle the English-versus-romaji question on its own.

I'm checking whether L3H4 reads something specific from its input, since a broad head that switches on foreign-letter statistics would write across many dimensions rather than just the weak ones.

I'm looking at L3H4's output, which seems to average MLP2 features across the context and push rare-letter logits, so it might track a foreign-word or language-mode signal. That would fit both English foreign names and romaji foreign passages, so it doesn't settle the question by itself. I'll also check whether L0H0 attends locally or mostly to itself.

I'm sorting the L0 and L1 heads by their attention patterns: L0H0 looks local, L0H1, H2 and H4 are position-modulated broad heads, and L0H5 is nearly uniform. For layer 1, I'm tentatively marking the heads as local, but I still need to check whether the positional term is actually large enough to dominate the content signal there.

I'm sorting the layer 1 and layer 2 heads by how their attention is shaped: their positional preference looks weak, so I'm treating the layer 1 heads as local, since they seem to favor recent other characters. For layer 2, I'm leaning toward local for most heads and broad for L2H3, though these labels are still uncertain.

I'm now sorting the layer 3 heads: H0, H1, and H2 look broad because their positional preference is weak and they still show a sink, while H5 seems local but weak. I'm not sure about H1 and H2, since their rotation angles could support either reading.

I'm labeling the L3 heads H0, H1, H2 and H5 as broad, since their attention seems driven by content features and a sink rather than by position. I'm less sure about H1, so I may mark it as unclear. Next I want to check the L1 and L2 heads by looking at whether their query-key matching on the token identity is positive or negative.

Next I'm writing the weight-scale observation, comparing the standard deviation of the query weights in layer 0 against the other matrices. After that I'll set the final probabilities: Python at 0.2, romaji at 0.22, English at 0.58, and the as-is corpus ordering at 0.85 versus 0.15 for shuffled.

Now I'm reading the L0H3 numbers: the QK matrix has a flat spectrum and its q and k share a roughly 12-dimensional subspace that barely overlaps its OV input, which fits a head attending to a fixed earlier offset and copying character identity. L0H5 looks different, with weak QK and strong OV, so it's probably averaging almost uniformly over the context.

Now I'm checking the L0 heads that feed the sink: the 2-d positional feature gets written into a handful of residual coordinates, and L0H1, H2 and H4 share that same input and output direction. Next I'm looking at how the layer 1 through 3 heads key-compose with those L0 heads, and whether a shared positional bias explains it.

Now I'm pinning down the L3H3 result: a positive query-token and key-L0H3 match, with a z-score of 12.4 against a random-rotation null, and the highest key-composition with L0H3 of any head. Its output writes heavily into the top direction of W_proj.3 with no sink bias, so I'm treating it as a copy head. I'm also noting that W_proj.3 concentrates nearly all its energy in about 28 directions, with the largest singular values dominated by the first one.

So the 28 output directions carry most of W_proj.3's energy, with four weaker ones coming in pairs that likely map to the rare letters j/x and q/z. That points to a 26-letter alphabet, and the layout fits English better than Python or romaji: a single late induction head rather than a stronger, earlier one, and no sign of the newline and indentation machinery Python would need.

