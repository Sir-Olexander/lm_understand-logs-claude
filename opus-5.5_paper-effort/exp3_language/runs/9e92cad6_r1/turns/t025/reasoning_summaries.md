Directions 0 through 25, 26 dimensions in all, are strongly enriched for the token reader at 2.5 to 7.8 times, with no positional signal, while directions 26 and up sit at random level for both. That puts the token subspace at exactly 26 dimensions. Directions 24 and 25 are clearly token-driven too, at 3.47 and 3.78.

Dimensions 26 and up come out at random level for both the token and positional readers, so the token subspace sits at exactly 26 dimensions. I'm now asking what 26 corresponds to: English/Python would allow up to 28 tokens, while romaji gives at most 24, so the count doesn't obviously match either. I'm also checking whether projecting out the attention-output subspaces could have cut into the token directions.

The per-vector mean gets subtracted, so the token embeddings' span only drops by one if it happens to include the all-ones direction, which is generally not the case. A shared mean direction adds the same vector to every input and works more like a bias than something the readers need to track. So the number of dimensions that actually vary across tokens is probably one fewer than the live token count, which would put 28 tokens at 27 dimensions and 24 tokens at 23.

The token subspace count probably isn't a clean match to either vocabulary, since 26 falls between what romaji (at most 24) and English/Python (up to 28) would allow. If the mean direction is absorbed as a bias, the varying dimensions would be one fewer than the live token count, which points to 27 tokens for English/Python rather than 23 for romaji.

I'm checking whether the 26-dimension count holds up under a different projection. I'll look at the V0/V3 cosines, which seem to knee around 24, and compare against the FC reader at 26. If the count stays put when I remove fewer attention-output dimensions, the English/Python reading looks more solid.

I'm going to check the L1 heads' Q/K readers as a second token signal, to confirm that directions 24 and 25 really are token-driven. Then I'm weighing whether 26 or 27 live tokens fits English or Python better, since the 27th dimension would need a weak token to account for it.

I'm checking whether romaji could account for this, since Aozora texts sometimes include half-width English or German words, which would add letters beyond the Hepburn set. That would push the live token count past what the romaji alphabet alone allows, so I'm weighing whether that's enough to explain a 26-dimension subspace.

I'm counting how many of the 24-ish live tokens would be frequent enough for English versus romaji to produce the observed dimensions. English has about 24 common tokens and four rare ones (j, q, x, z), while romaji is similar but also needs l, q, v, x. Both come out close, so the frequency count alone doesn't separate them.

I'm now comparing letter frequencies to see whether English or romaji fits better: romaji has a frequent core of about 24 tokens plus a few moderately common consonants like j, c, f, and p, with l, q, v, and x nearly absent. Python would have all 26 letters at 0.2% or more, plus space and newline.

I'm checking whether the 26 strongly token-driven dimensions could correspond to English or Python rather than romaji, since English's rarer letters might each carry a weak direction of their own. Romaji would leave the l, q, v, and x directions nearly empty, so the 26 dimensions point toward a richer alphabet. The 25th and 26th dimensions still look clearly token-driven, which keeps 27 live tokens as a plausible fit.

I'm weighing the remaining token dimensions against vocabulary sizes: the rare letters in English (q, z, x, j) might show up as weaker dimensions, while Python's 28 robust tokens would predict 27, and romaji's roughly 24 would predict 23, so 26 sits closest to English.

Python and English both fit the 26-dimension count better than romaji, so I'm setting romaji aside for now. Next I want to test whether the count holds up: I'll vary the projection size, try the layer-1 heads' Q/K readers as an alternative token signal, and check whether MLP1 also reads the raw token directions.

