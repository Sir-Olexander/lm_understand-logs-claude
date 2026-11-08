I'm finding that the attention heads in the later layers rely heavily on the S-channel: layer 2 across all heads, and layer 3 heads 2 through 4, while layer 3 heads 0, 1, and 5 don't. That makes the S-feature look like a broad segment or position signal rather than something narrow, and I'm starting to wonder what that means for the keys.

I'm weighing whether the 53° plane is a slow positional signal, which would make the L0H3 attention recency-like, or a fast rotation that would make it periodic. I'm also considering L0H5, where the query and key vectors overlap heavily, as a way to pin down the sign of the S-feature.

I'm checking whether L0H2's 52.6° rotation in the slow plane would add a broad bias toward keys about 37 positions back, which seems at odds with it being a sharp prev-token head. Maybe the high-frequency planes dominate there, or the radius in the slow plane is small. I'm not sure the positional embeddings are even identifiable this way.

I'm working through how the input side of the L3H1 induction head reads from the MLP0 and attention outputs, and wondering whether that enrichment is meaningful. I'm also considering a different counting route: L0H2's OV mapping token embeddings forward into L1 queries, which would be a bigram signal. Rare romaji tokens like l, q, v, x still have near-random embeddings, but I'm not sure the layer norm and positional embeddings wouldn't swamp them anyway.

I'm turning to whether the token counts can be inferred from corpus-level statistics, such as how predictable each text is and how much structure its positions carry. Python's indentation and line-position patterns look like a different signal from English's word-level structure, but I doubt any of these are calibratable from the weights alone.

I'm thinking about which kind of corpus would lead the first layer to devote four of its six heads to segment-style features while only one acts as a previous-token head. Maybe the answer is something like word or line boundaries, since Python relies on both indentation and line structure. I'm not sure the weights alone can tell me which corpus that is.

I'm now wondering whether the S-heads' output on dimension 222 is a massive-activation signal, like the attention-sink dimensions in GPT-style models that mark the first token. If the 53° plane carries a large component for position 0, that would explain why the S-heads write almost entirely to that single coordinate.

I'm now thinking the sink idea fits: the S-heads average a position-0 signal into dimension 222, giving something like a 1/(i+1) scalar, and the later keys read that dimension to pick out the first token as a no-op target. The query's bigram dependence would control how strongly each head pulls toward that sink.

I'm now treating the sink idea as the likely explanation: the S-heads average a position-zero signal into dimension 222, and the later heads' keys read that to send attention to the first token when they need to do nothing. The exact content selectivity of those heads is less clear to me. For the head labels, I'm leaning toward calling L0H0, H1, H4, and H5 broad, since their QK structure is low-rank and low-norm.

I'm checking the L1 and L2 heads next: their positional parts are small, and their keys seem to read the same S-channel, so I'm not sure whether they should be labeled broad or local. For L3H5 I'm guessing broad, since its QK is small but its OV is large, and L3H0 and L3H2 through H4 still don't have a clear label. I want more evidence on how the L1 through L3 heads match content, so I'm looking at their QK structure.

I'm checking whether the L1 heads' negative query-key structure makes them attend away from the current position, so they'd pick out the previous tokens and build a trigram signal. I'm also wondering whether their rotated positional components in the high-frequency planes could encode an offset.

I'm comparing the induction-head evidence across corpora: I'd expect Python-trained models to show strong induction heads, but here only L3H1 scores clearly, with its keys reading the previous-token signal only weakly. I'm also looking at the L1 queries, where about 18 to 25 percent read the previous token, which suggests their attention depends on the current-plus-previous bigram, and that pattern seems common to any corpus.

I'm weighing whether the space token's frequency could be read off the MLP0 and the T subspace, since spaces should be the most common token in any corpus and even more so in Python with its indentation. I'm also looking at whether the high-magnitude layer-norm dimensions line up with positions the attention heads read from.

I'm thinking about estimating the number of distinct tokens from the singular values of the L1 query matrices restricted to the prev-token subspace, though that doesn't look conclusive. I'm also considering whether the singular value spectrum of W_proj.3 could serve as a rough token count, so I'm looking at how the values fall off step by step.

I'm weighing the singular value spectrum of W_proj.3 as a rough token count: the drop-off suggests about 23 strong directions plus a handful of weaker ones. If the corpus is romaji, that would fit 24 used tokens with four unused ones, though the push-down directions for unused tokens could mix in any way, so the count stays uncertain.

I'm checking the romaji-versus-English count: a romaji corpus would leave only about one medium direction, but the spectrum shows five, which fits English's five rare letters and newline better. I'm now setting up the Python case to see whether its indentation-driven token mix would produce a different number of medium directions.

I'm now estimating the Python case the same way: space at roughly 20 to 25 percent and newline near 3 percent, with the letter frequencies from identifiers and keywords. That gives about 25 frequent tokens and three rare ones, q, j, and z, so I'd expect around 24 strong directions and 3 medium ones.

I'm now estimating the romaji case from letter frequencies: vowels like a, o, i, and u dominate, with n, k, and t close behind, and rarer letters like z, b, c, j, f, and p at the bottom. Space would sit around 15 percent, depending on how words get segmented, and the newline count is still unclear.

I'm comparing the singular value drop-off to the English and Python frequency estimates: English fits about 22 to 23 strong directions plus 5 medium ones, while romaji fits worse. I'm now asking whether a frequency threshold actually determines strong versus medium, since the spectrum of W_proj.3 reflects how the MLP output distributes logit variance across tokens.

I'm weighing whether the medium singular values (around 8 to 7) reflect rarer tokens, roughly a tenth of a percent of text versus about one percent for the common ones. I'll test that idea on other matrices, such as the L3H1 copy-head OV, which reads from the input token space.

I'm comparing the copy head's strong singular count, 24, against W_proj.3's 23, and both sit near the frequent-token counts I estimated for English and Python. The count still doesn't clearly separate the three corpora, so I'm turning to the input side with L0H2 next.

I'm checking whether the indentation count could show up in the prev-token head's decay curve, but that doesn't seem to separate the corpora. I'm now considering a head that attends back to the start of the previous line to copy its indentation, since Python's runs of spaces in multiples of four would make that useful.

I'm now asking whether the position-in-line matching idea holds up, since the L1 queries read the position feature only weakly, which would argue against copying indentation that way. Another possibility is that a head attends back to the newline before the previous line and counts spaces from there. I'm also thinking the space and newline identities should show up in the L0H2 previous-token output that the L1 queries read, so I'm going to check that next.

I'm looking at the final layer's most-shared output direction, #1, which several heads write to along with MLP0, and wondering whether it encodes something like the space-versus-letter distinction or the newline. The broadly attending L3H5 head has a large OV norm but only a small share of its output lands in the logit space, with its main contributions going to directions #1, #24, and #29 through #31.

I'm weighing a broader idea now: a head with large OV output but diffuse attention might encode a context-level statistic, like whether indentation or newlines appear nearby, instead of identifying one token. Anchoring tokens from the weights alone is hard, so I'll likely settle for probabilistic guesses on the corpus. Python's indentation-driven structure over the 128-token window is the feature I'll test next, along with whether English's word and sentence patterns would show up differently.

I'm checking whether induction strength tells the corpora apart: L3H1 is a real induction head, which fits English-style character data, though a romaji corpus with repeated names and phrases could produce the same pattern. That makes the induction signal a weak discriminator, so I'll look at the broadly attending heads in L3 next.

I'm wondering whether a head that attends to the previous line's start could copy indentation, but the query-side position signal is weak, which argues against that. I'm also considering that "space after space" never shows up in English or romaji text, while Python has it only at line starts, so the model would need to learn to suppress that pattern differently in each corpus.

I'm considering whether the token embeddings themselves carry the frequency structure, since rare letters should sit apart from the common ones in the embedding geometry. If I could read off the Gram matrix of those vectors, the vowel cluster and the outlier 'n' might show which corpus this is, but I don't have the token vectors in hand yet.

I'm wondering whether the S-channel is a sink-like feature, since if the S-heads respond mostly near position 0 their fixed key preference would point that way, but I can't check that without position information. I'm also considering how the benchmark designers likely framed the task: the cues they'd expect are induction heads and corpus-specific signals, and the question is which of those the weights actually show.

I'm weighing whether a head that counts spaces from the last newline could explain the indentation, since Python would need that for columns. English would have little reason to devote heads to newline tracking, so that's a possible discriminator.

I'm checking L0H5's query and key directions, which overlap at 0.74, so the head seems to match keys that look like the query in that feature. But the plane numbers for the query and key don't obviously agree with that overlap, so I'm trying to work out which measure I'm misreading.

I'm working out whether the rank-1 QK score range depends on how the token and position embeddings spread their energy across the dimensions, since the normalized input should have a fixed total norm. If that energy is spread out, the scale of u's projection onto it determines how sharp the attention is.

I'm wondering whether the positional embedding is what fills that 53° plane, since the top dimensions there are all ones with large layer-0 norm weights. If that's the case, the query dot products could be large enough to sharpen attention on their own, and I'd like to reconstruct the position signal from L0H2's sharp previous-token behavior to confirm.

I'm thinking about how romaji differs from English here: syllable structure makes each character depend mostly on the one or two before it, while English leans more on whole words. Short particles like "wa", "no", and "ga" would make word-level features matter less for romaji. For Python, both line-level and word-level structure seem important.

I'm looking at the layer-norm weights around dimension 222, since the S-heads push a large single coordinate there and the question is whether later layers amplify it. L1's attention norm weight at 222 is about 1.12, well above its average, while MLP0 largely ignores it. I want to check the same dimension in the other layer norms, along with 78, 360, and 72, before drawing any conclusion.

I'm adding dimension 103 to the list of layer-norm weights to check alongside 222, 78, 360, and 72.

