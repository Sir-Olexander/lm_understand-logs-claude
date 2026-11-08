I'm noting that several heads across layers 1 through 3 share the same f-plane structure, with rotation around 72 to 75 degrees and a query-independent preference for keys pointing along the (-1, -0.7) direction. Now I'm turning to the rotation.

I'm working out whether this shared rotation acts like a generic sink or recency bias, since the same angle recurs across many heads. If the positional features trace a spiral with phase growing over time, a fixed rotation would favor keys at a fixed lag, which points toward attending to nearby previous positions. The per-head content differences still need separate checking.

I'm checking whether the L1 heads act as local, relative-position heads, with L1H0 maybe broader given its weaker rotation and small magnitude. L2H2 looks like a previous-context head despite its small f-plane fraction. Next I want to test whether the L0 heads that key on spaces can separate Python indentation from English word boundaries.

I'm weighing whether the L0 heads keyed on spaces can tell Python indentation apart from English word boundaries. The thought is that counting indentation might rely on tracking the position since the last newline, which L0H3 could be doing if the newline token is the one it keys on. I'm still fairly comfortable putting Python at around 0.55 to 0.6.

I'm ruling out romaji, since the 27 or 28 strongly read dimensions would need letters like l, q, v, and x to be meaningful, and that seems unlikely. Between English and Python, I'm leaning toward Python based on how the densest MLP0 token has its own doubled-token neuron cluster, though I'm still working through that part.

I'm weighing whether the doubled-u neurons point toward whitespace runs in indentation, since English letter doubling would likely spread across many letters rather than cluster on one token. Against that, English would make L1 a local-context head rather than an induction head, though that evidence is weak.

I'm checking whether the doubled-u cluster really stands out, by comparing how many neurons show strong self-similarity for u against the other letters. If it's concentrated on whitespace runs, that would fit indentation far better than English letter doubling, which tends to spread across many letters.

I'm weighing romaji against Python in terms of doubled letters: romaji would have many spread-out doublings like "nn", long vowels, and "tt" or "kk", while the single dominant u cluster fits Python whitespace better. I'm settling on Python near 0.6, English around 0.3, and romaji around 0.1, and I'm still deciding whether to check the shuffled baseline.

I'm checking whether the as-is reading holds up against the shuffled baseline, since adjacent spaces would also show up there and could blur the doubling evidence.

I'm checking whether the shuffled baseline would produce the same shift-aligned matches, and I don't see how it could, since those patterns depend on the real token order. I'm treating the as-is reading as the working conclusion. I want to run a few more checks on the remaining L2 and L3 heads to firm up their classifications.

I'm checking whether these heads are doing copy-style attention through some other carrier, since they don't look like induction heads. I'm also comparing their same-token and positional-key signatures from the earlier run to see which ones lean toward the positional features.

I'm grouping L2H0, L2H3, L2H4 and L2H5 with the L1 heads as positional, local-style attention, since they avoid same-token matches and rely on the f-plane mechanism. L3H2 and L3H4 show weak signatures I can't place yet, and L3H3 looks broad because of its weak QK but large OV contribution.

I'm weighing the L0 heads that key on spaces, but the remaining question is whether L2H0 and L0H2 are really doing anything distinct. L2H0's small positional share and negative same-token signature leave its role unclear. L0H2 has weak QK but a clear token preference for whitespace, so it looks more like a broad, token-driven head than a sharp one.

I'm checking whether the second-densest MLP0 peak shows the same doubling pattern and how it connects to the L0 heads, since that could help tell Python whitespace apart from English letters. I'm also weighing whether a triple-space cluster would show up in the offset-2 head, which seems odd if the indentation story is right.

I'm testing whether the L0 heads that key on spaces could be counting indentation: if three of them attend to space tokens with positional weighting, that would track how many spaces came recently, which fits Python better than English word boundaries. I'm still leaning toward Python at around 0.55, and I want to check whether the L0H4 value output actually reads the space direction or a different peak.

Now I'm weighing the L0 heads' OV outputs, and I'm settling on the final numbers I want to cite. The u-token weights on L0H0 and the peak-2 weights are close enough that I won't read much into their ordering. I'm collecting the key values now: the bimodal ln1 weights in the positional subspace, and the L0H4 and L0H0 QK and OV magnitudes.

Now I'm checking the layer 0 heads, where L0H5 has most of its query-key weight in the positional subspace and its output reads the token features with a fairly flat spectrum. The token dimensions shared by H1 and H5 look like they carry the 28 symbols the MLP0 neurons use, and layer 1 seems to pick up on those through its key composition.

I'm now going through the layer 2 and 3 heads' scores: the induction signatures look strong in L2H1 and L3H5, with L2H2 showing a similar shifted pattern, while the reverse alignments come out near zero. I'm also noting that the densest MLP0 group has 9 of its 34 neurons reading the same token at the previous position.

I'm settling on the final write-up now, citing the key numbers: the MLP0 token that doubles its detectors is also the key preference of L0H0, L0H2 and L0H3, and the OV traces across layers 1 through 3 are all negative.

