I'm checking the singular values of the induction head's OV map: about 25 are significant, with the 25th at 2.03 before dropping off sharply. With kT=40 and kL=28 the spectrum declines smoothly, and by the 25th value it is around 0.39, so the copy map looks like it has an effective rank near 24.

I'm going back and forth on whether the rank of 25 fits a 27-dimensional logit space or a 23-dimensional one. The U_28 restriction might not match the logit subspace exactly, so I want to check how the rank grows across U_24 through U_40 and whether it saturates.

I'm weighing whether the rank of 25 could come from rare tokens like 'j' or from the romaji letter set, which would give about 23 logit dimensions and so doesn't quite fit. The gap might instead come from non-token directions such as a confidence or normalization direction, so I'm looking for a way to tell those apart.

Now I'm checking whether the token-specific part of the induction head's output fits romaji or English, since the common directions could account for a few of the 25.

I'm weighing whether the spectrum of the same-token circuit can separate romaji from English, since doubled vowels and consonants are common in romaji but rare-token singular values are too noisy to tell them apart cleanly. Doubled letters in romaji come from long vowels and sokuon, which may leave a distinct trace in the attention pattern that I'd want to look for.

I'm considering whether the newline frequency could be the token type behind this structure: Python windows contain far more newlines than English or romaji text, and the line-start signal (indentation, leading keywords) would need its own positional copying. If that's the case, the copying heads would be tracking where the current line begins.

I'm now checking whether the L0 heads' key preferences point to newline or space, since a dedicated newline-position signal would only be worth three heads in a corpus like Python. I'm also considering whether L0H1, H2, and H4 are tracking the start of the window through positional keys, or whether their token-based component is doing the real work.

I'm now wondering whether the induction head's strength itself gives a clue about the corpus, since Python code repeats names like "self" and function identifiers far more than prose does.

I'm now checking whether the induction head's strength says anything about the corpus, since Python repeats identifiers like "self" far more than prose does. The model would likely build earlier, stronger induction heads in that case, but only one sits in the last layer here. I'm also thinking about whether a line-level copying structure would show up in these heads.

I'm looking at how the induction head matches on longer context, since its query combines the current token, the previous token, and layer-2 features against keys built from earlier tokens. Looking at the layer-1 and layer-2 contributions, this looks like typical multi-token copying. Now I'm thinking about whether romaji's consonant-vowel alternation would show up in how the head predicts the next token.

I'm weighing the singular-value spectrum for the vocabulary estimate, and I'm going to look at the layer-0 MLP and the prev-token head restricted to the live tokens, to see whether a cleaner count comes out than the fuzzy rank so far. The corpus question stays open, since the evidence points loosely to about 24-25 dimensions and doesn't settle which language it is.

I'm now asking which characters the romaji text never uses, since letters like l, q, v and x would leave their embeddings untouched. If the live token set is about 24 letters, that would account for the rank near 24 I've been seeing. Rare tokens in the input get small singular values because the gradient is weighted by frequency, which makes the count fuzzy.

I'm working through which corpus fits: English has about 28 live tokens, romaji about 24, so the tail of the singular-value spectrum should stop at a different index in each case. The rare letters are where the spectrum thins out, and the values from V3+V0 after removing P, running down to about 0.29 before going to zero, are the numbers I need to compare against that.

I'm checking the singular values past index 22, where they settle near 0.2 as a noise floor, and I'm wondering whether the count of about 24 token dimensions holds up, since English would have about 28. Two separate measures both point to roughly 25 or fewer, but I'm not sure that's decisive yet.

I'm trying to get a cleaner token-dimension count by reading the token signal from more places: layer-0 values, the layer-1 projections, and the MLP0 input. After stripping out the positional component and the attention outputs, I'd only use the layer-0 readers, which see just token and position.

I'm now trying to separate the token and positional directions by a joint factorization of the reader matrices, since the earlier estimate of P depended on which readers I included.

I'm setting up a generalized eigenproblem to separate the token subspace: directions that V3 and V0 read but Q3, K3, and V5 don't. The ratio of read strength to positional reading should show a gap, and counting the high ratios would give the token dimension.

