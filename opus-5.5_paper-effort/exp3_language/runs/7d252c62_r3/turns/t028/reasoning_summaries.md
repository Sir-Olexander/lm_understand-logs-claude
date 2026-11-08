Baseline |e|/sv sits at 0.27 for random independent pairs. The L2 and L3 heads hover around 0.28 to 0.33, close to random, which suggests their Q and K read mostly different subspaces, with L3H5 the exception at 0.55 and a trace of +0.49 pointing to same-feature matching. L0H1 and L0H5 are high at 0.79 to 0.84, meaning both sides use the same positional subspace, and L1 sits near 0.4. For the OV traces, every head in L1 through L3 comes out negative, roughly -0.2 to -0.29.

Now I'm checking the negative OV traces, which all sit well outside the random baseline. L3H5 comes out at -0.39. Almost every head in L1 through L3 shows this sign, which seems odd. Positive traces are what copying heads usually show, so I'm wondering whether LayerNorm's mean subtraction is behind it.

I'm weighing whether a bag-style head would mainly track the line's overall character composition, which would push the trace positive, or whether consuming already-seen characters matters more in short Python lines. I'm also considering that in ordinary text, negative OV might reflect suppressing repeats, like not continuing "ll" into "lll."

Now I'm looking at the OV eigenvalue spectrum for L3H5 and the other heads, to see where the negative directions sit in the residual basis. I'm also weighing whether the embedding could be reconstructed from the constraints, though the unembedding is unknown, so that route stalls. Next I want to compare attention weight norms against the MLP weight norms across layers.

I'm wondering whether a positional kernel is the clearest way forward, since L0H4's uniform averaging could help separate position effects from the L1 heads. I'm also weighing whether the head's keypref bias explains the small L0H1 value, which points toward a weak absolute-position preference.

I'm weighing whether L0H1 and H5's shared key-token direction is a positional head that also keys on a single token, such as attending to the previous token unless it's a newline, or to the most recent space. The key-token preference is weaker than the positional part, with singular values of 0.85 and 0.68 compared to roughly 4.7 and 3.2.

I'm thinking about the layer-0 heads in the shuffled setting: one could find the last newline by weighting keys toward newline tokens and recency, then copy that position as a line-start marker for layer 1. In the as-is setting, the layer-0 heads would more likely handle previous-token, recent-token, and newline or space detection. Here H1 and H5 do positional work with some token copying, while H4 and H0 average uniformly and copy positional information.

The uniform heads L0H4 and L0H0 seem to copy positional information but not token identity, so they don't look like they're building a bag of recent characters. If the shuffled-text model needed a window bag, I'd expect one of them to copy tokens, so maybe that job falls to L1 instead. For the shuffled Python case, I'm guessing the line-start detection and H1/H5's local-window behavior matter more than the uniform heads.

I'm weighing whether M5's larger rank, around 20 components, marks it as a sharp previous-token kernel, since a sharp offset needs many frequency terms while a smooth window needs few. M1, with its dominant low-rank structure, would then be the smoother one, though its 180° component complicates that reading.

Now I'm comparing the angles of M5's components against M1's, testing whether M1 might be a two-step offset kernel. If that held, the predicted angles would be roughly double M5's, and the observed values don't line up cleanly.

I'm comparing the angles against M5's predicted ones under a few offset hypotheses, and the high-angle components line up reasonably with an offset near 2, while the low ones and the 180° component don't. The largest component only fits an offset of about 1, so a pure offset-2 reading doesn't hold for M1.

I'm testing whether the big M1 component could be a frequency-like term rather than an offset signal. If both heads were offset-1 kernels, their component angles should match, and they don't. The offset-2 reading fits the higher-angle components but breaks on the large one near 58° and on the 180° component.

I'm testing whether M5 is an offset-1 kernel and M1 an offset-2 kernel, comparing each M1 component's angle with twice the matching M5 angle. The higher-angle components fit reasonably, but the low ones and the 180° component don't, and the systematic excess of roughly 15° is the part I need to explain.

I'm checking whether the 2-and-3 offset mix fits the low-frequency components of M1, but the weak eigenvalues make those angles unreliable. The 180° component looks like an offset-2 signature for M1, while M5 at 180° fits an offset-1 kernel.

Checking the period-2 idea against M1's negative real eigenvalue at 180°: a plain offset-2 kernel would give a positive term there, so that doesn't fit. A slow one-dimensional component with a negative sign could, though, which would mean the 180° piece isn't part of the offset reading at all.

I'm working through whether the positional embedding can be reconstructed by assuming it is a sum of rotating components with unknown frequencies and amplitudes. That would mean fitting the layer-0 score kernels directly, but without any ground truth to check against, the fit has nothing to anchor it, so this route looks unlikely to pin anything down.

I'm looking at the layer-1 heads now: they all seem to attend based on the position code that L0H4 writes into the keys, so their QK structure looks alike. I'm also noting that the L1 through L3 heads have negative OV traces, and that L3H5 looks like a symmetric matching head that attends to positions sharing its features and subtracts them. For shuffled text, I'm guessing the layer-1 heads would build a bag of the current line by attending to positions inside it.

I'm noticing that all the layer-1 heads key on the same positional code from L0H4, which fits a positional-range pattern rather than content lookup. The L1 keys look about 95% positional, so I don't see evidence of content-based induction-style attention there.

I'm weighing whether the negative OV traces in these later heads fit a consumption pattern: a head attending to earlier characters in the current line and subtracting their identity would lower the chance of repeating letters already used. In as-is text, I'd expect the copy-style heads to instead add their attended token, so the sign would flip.

I'm checking whether the negative OV traces make sense at all, since copy-style heads should usually add the attended token. Restricting OV to the token subspace, T^T OV T, should show whether these heads copy identity positively, though the final output projection is unknown, so that only partly settles it.

I'm checking whether L3H5's query and key features line up with token identity, so I can see if its negative OV output on those same features fits a repetition-suppression reading. The obvious candidate for as-is text is a head that copies what followed earlier occurrences of the current token, but a causal key can't store the next character, so that copy story doesn't work for this head.

I'm wondering whether a head attending to earlier same-character positions in the current line and writing a negative signal could track how many of each character are already used. But the next character in shuffled text is independent of the current one given the bag, so restricting attention to matches of the current character seems off; a bag-style head would need to attend across all characters instead. The line-start idea I was about to explore is still open.

I'm checking whether the L3 heads behave like induction heads: the key would need to carry the token before its position and the query the current token, with the OV copying the key's own token. The QK cross-subspace numbers near random make that possible but don't confirm it, so I need some other anchor to decide.

I'm checking whether the L2 and L3 heads read the current token through Q and the previous token through K, since their enrichment ratios are moderately above random. I'll compute the block norms of the token-to-previous-token interaction in both directions to see which way the signal runs.

I'm printing the full enrichment matrix now, pairing each of the current-token, previous-token-via-L0H1, and previous-token-via-L0H5 features on the query side against the same three on the key side for L1 through L3, to see which direction carries the induction-style signal.

