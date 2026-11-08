I'm checking how the eigenvectors of C line up with the token set T and the set P: most of them aren't fully inside either, so the rest must sit in other parts of the 384-dimensional space. Restricted to T, C has roughly 18 positive eigenvalues, about 14 of them significant, which points to L3H1 attending from the query token to the key's previous token, so it looks like an induction head. I also want to look at the layer 1 and 2 heads as possible previous-token sources.

I'm now weighing whether the induction-head reading holds up against shuffled data: the L1 and L2 heads could also be carrying previous-token information, and I don't see a simpler alternative like pair counting that would produce the same query-to-previous-key pattern. Turning to the corpus side, I want to know whether the L0 heads are tracking spaces or newlines, which would decide whether they encode word or line positions.

I'm working out what L0H0 would be doing: if its key side picks out spaces, it would attend uniformly to every earlier space, which gives little positional information, and the same holds for newlines. Unless the query-position factor turns negative at some positions and makes the head avoid certain tokens, neither reading looks especially useful.

I'm now checking whether the query-side partner of each layer 1 head reads position or token information from the output direction of L0H0. If the resulting score sits mainly in the position subspace, the head would be doing recency-weighted lookups, and if it lines up with a shared constant direction, it would just be a key bias. I can't pin down that constant direction without the average input, so I'm stuck on how to estimate it.

I'm now going to check a different probe: the copy mapping of L3H1 into the output space, to see how many tokens it actually copies, and then fold that into the induction-circuit picture I've been building.

I'm weighing what the corpus might be from the surface features: romaji would show a tiny syllable inventory with frequent particles like "no", "wa", and "ga", while English prose has a much richer vocabulary and long paragraph lines. Python would stand out through keywords such as "def", "self", and "return", plus indentation and heavy repetition of "self", which is where I'd expect induction to matter most.

Now I'm checking L3H5, which has weak query-key scores but very large output singular values, so it may be doing broad averaging over the context rather than a sharp copy.

I'm trying to tell which token L0H0's key side selects, using the other token-specific circuits as a reference, since a newline or space head would behave very differently in the attention pattern.

I'm now checking whether L0H0's key-side preference is a sink-like token that the head attends to broadly, and whether its positional output carries any useful information. Next I want to see if the layer 1 LayerNorm bias gives large constant components to the query and key, since a constant query would act as a plain key bias that reads the broad L0 output directions.

I'm checking whether the constant part of the layer 1 query, pushed through the key weights, gives a query-independent bias that reads the broad layer 0 output directions, which would act as a recency or line preference. I also want to compare how much line structure would matter for distinguishing Python, romaji, and English prose.

I'm checking whether the L0 heads' key-side features are tied to newlines or spaces, since that decides whether D tracks line positions or word positions. If the broad heads write the position of the last separator before each key, the query-side score would be position times that separator position, which would suggest a recency-style lookup.

I'm considering whether the broad L0 heads' query factor changes with position, maybe sharpening later in the sequence to offset the growing number of keys. Since nothing here is recency-based, the head is likely averaging over every matching token, which would make its positional signal roughly the mean position of the matches seen so far.

I'm checking whether the broad layer 0 heads might just be producing smooth positional features, with the token preference as only a minor modulation. Comparing the token-side key directions of L0H0, H1, H4, and H5 shows they aren't the same direction, so I'm wondering whether anything in the weights could fingerprint the corpus, starting with Python's indentation structure.

I'm weighing whether the spaces-and-underscores normalization of Python identifiers would shape the model's prediction of leading-space runs after a newline. I'm also considering whether romaji's strict consonant-vowel alternation would lean more on the previous-token head than English does, and I'm wondering if the model's long-range versus local reliance could separate the corpora.

I'm checking whether the MLP0 dominant input direction reads a token feature, like "is space," or a shared position or line signal that recurs across the layers. The same direction seems to dominate each MLP, so I'll compare its overlap with the D, P, and T subspaces to see which one it lines up with.

I'm weighing which weak cues might tell the corpora apart, starting with the token inventory: about 24 strong dimensions could fit romaji or English, but Python's common letters would likely push it a bit higher. I'm also comparing letter frequencies in English prose to see how that inventory would look.

I'm listing the letter frequencies for English and romaji, and noticing that the rare letters j, x, q, and z show up at only about a tenth of the rate of v.

I'm weighing the letter inventory for Python against English and romaji: Python's identifiers pull in x, k, v, and j, and at roughly 0.3% each for q, z, and j, none of its letters drop below about 0.2%. So the rare-letter gap that separates English from romaji would probably not appear in Python.

I'm comparing the spectra now: English would leave about four medium-strength letters (j, x, q, z), romaji would leave almost none of those, and Python's letter mix has no letter that rare. The observed output directions show a clear strong block of about 23 to 24 dimensions followed by a band of medium ones.

Now I'm weighing whether the extra dimensions in the output spectrum come from a romaji or English token set, or from directions that just control the final norm. If it's the norm, the 27–28 count tells me little about the corpus.

I'm weighing whether the extra dimensions in the output spectrum come from a romaji or English token set, or from directions that just control the final norm. The cluster of five medium-strength values after the clear block of about 23 dimensions could be four rare English letters plus one other feature, which would make the count less decisive than it first looks.

Now I'm working through the rough romaji letter frequencies, with vowels like a, i, and o at the top and k, s, and t in the middle, and I'm weighing whether p and f would count as rare enough to match the English pattern.

I'm checking how the rare letters would show up under Python's identifier vocabulary, since q, z, and j sit around 0.3% there, which is far above the rare-symbol gap I'm looking for. That would make the medium-strength band fit Python less cleanly than English.

Now I'm weighing the medium-strength dimensions in the L3H1 copy output: 24 strong ones, with the rare letters j, q, x, and z likely contributing the weaker band, which would fit English better than Python.

I'm weighing the token-count cue: the six medium-strength dimensions in the L3H1 copy output fit English better than romaji, since romaji would leave only about two rare letters (f, p) to account for them. Python's identifiers would push toward more strong dimensions, so the count leans English.

I'm now comparing the rest of the symbol inventory against Python's: it has plenty of common symbols like space and newline, so the count of strong dimensions there would look different from the English pattern.

I'm comparing the strong-dimension counts against each candidate symbol set, and the English inventory (23 common symbols, 5 rare) fits the observed 23 strong plus 5 medium cleanly. Python fits less well, and romaji would need more dimensions than its symbol count supports.

I'm now checking whether the medium-strength directions in W_proj.3 are the same ones L3H1 writes when it copies rare letters, since that would tie the extra dimensions to the token inventory rather than to the norm.

I'm checking whether the copy head's output along each of those W_proj.3 directions is strong, which would tie the medium-strength dimensions to rare-letter logits rather than to the norm.

