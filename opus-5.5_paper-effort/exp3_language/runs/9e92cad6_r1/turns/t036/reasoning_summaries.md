I'm checking the top output directions for head O3: the V0-to-V3 enrichment is around 1, so none of them are token-related. The weaker dimensions 19 through 30 show QK enrichment in the 3 to 6 range, which points to positional dimensions. That would mean OV3's weaker components sit in positional space, possibly overlapping the layer-0 Q/K positional reading subspace.

The leftover fragment finishes the point about OV3's weaker components sitting in the positional subspace read by layer-0 Q/K, and that this implies OV3 writes positional-like vectors. I'm then questioning why removing O3's dims 25 to 30 cut the T-enriched count from 31 to 26, since O3's dims 13 to 25 aren't token-enriched. My working guess is that the SVD of FC0 times the projection mixes token and positional dimensions when their singular values are close, spreading token enrichment across more directions.

I'm working out why the count fell from 31 to 26 when the large-ks dimensions were removed: the cleaner 26 seems more reliable than the inflated 31, since mixing of token and positional directions in the SVD could be pushing partial token signal into extra directions. Checking the canonical correlations between the projected FC0 rows and the V0V3 subspace, the knee sits around 23 to 24, which puts the token dimension at roughly 24 to 26. Weighing the tokenization counts, romaji at 24 tokens fits that range, while English at 28 tokens leaves room for the rare-token adjustment.

Checking the English, Python, and romaji token counts against the observed 26: English at 28 tokens with a few rare ones fits best, and the LN mean direction could be the next thing to check.

I'm checking whether the FC0 spectrum keeps showing token signal past the 26th direction. The 27th has a weak token enrichment of 1.11 and a singular value of 4.46, which could be a faint token direction mixed in or just noise, so I'm going to look at the spectrum beyond 26 with a T-enrichment computation.

I'm weighing which language the model most likely was trained on, using the token-count fit: English at 28 tokens matches best, so it's my leading guess at about even odds. Romaji and Python each look less likely, with Python having a slight edge from the induction-head argument and romaji's 24-token count sitting below the observed 26.

I'm now asking why all six layer-1 heads strongly avoid attending to identical tokens. That's expected for Python, where long runs of indentation spaces would make them look back past the newline, but it's less obvious for English, where same-token avoidance should matter less.

I'm now checking whether L0H5's third OV direction, which reads token identity through a window-level bag, lines up with one of FC0's top directions. I'm also wondering whether window-level statistics would matter more for Python's indentation density or English's rare letters, though this single dimension alone won't settle it.

I'm looking at layer-0 heads 1, 2 and 4: their token preference is only about 0.3 to 0.4, close to what chance would give, so they're mostly positional. A strong newline-keyed head would be expected if line structure mattered for Python, and its absence points me toward English or romaji over Python.

I'm checking whether any later-layer heads show a strong token-type preference that doesn't depend on the query, by looking at the key's token-direction component against the constant query bias.

I'm now checking whether any heads in layers 1 through 3 prefer a particular token type no matter what the query is, by measuring the token-direction part of each key against the constant query bias. A head with a strong, query-independent preference would be a line-structure head, the kind I'd expect if newlines mattered for Python.

