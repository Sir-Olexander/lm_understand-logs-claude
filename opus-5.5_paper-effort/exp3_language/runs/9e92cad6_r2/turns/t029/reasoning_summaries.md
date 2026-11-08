The MLP outputs seem to barely drive the later heads' queries and keys: L2 and L3 read them at about baseline, and while values do read them more, that doesn't explain the attention patterns. So I'm wondering whether L2/L3 Q/K are instead reading the outputs of L1 and L2 attention.

I'm checking whether L3H3's Q/K read anything specific: the residual inputs from the token embeddings, S3 output, layer 1 and 2 attention outputs, and MLPs all come out at moderate, similar levels, so it looks like broad residual similarity rather than any one feature. I'm also wondering whether the remaining unexplained residual dimensions are token identity or noise.

I'm settling on the shuffled-text hypothesis, since L3H3 looks like a content-similarity head over the whole window, which fits positions sharing a similar context bag, such as the same line. On the corpus, the token dimensions point toward Python over English over romaji, and the line-tracking machinery across the early heads supports short lines, which favors shuffled Python as the main candidate.

I'm working out what a shuffled-line model would need to predict the next character: the line's remaining characters are whatever is left of its multiset, so the line-so-far bag, its length, and the earlier lines matter. The recent few characters from the local head only give a partial bag, and that same head also signals when a newline just occurred. Still, a sharp local window over the last few characters would probably need a high-rank, multi-frequency positional pattern.

Weighing a final consideration: the group newline heads and the negative OV traces both fit short lines in shuffled Python, which suits the earlier line-tracking evidence. English shuffled would leave those mechanisms with little to do, so Python ranks above English under the shuffled hypothesis. I'm putting the as-is case at about 25%, with the corpus split roughly Python 0.55, English 0.3, romaji 0.15.

I'm checking whether L1's recency-style heads might be previous-token heads via a recency bias plus self-avoidance, which would make L1 a local n-gram gatherer like L0H3 and L2's local heads. I'm also keeping in mind that a 4-layer character model with a 128-character window would normally develop induction heads for repeated names, so the lack of a strong induction signal still tilts me toward a non-as-is corpus.

I'm now checking L1's heads: the query-token-to-key-token term is strong in H0 and H2, while the summary-to-summary term is weaker, so the mix is uneven. I'm also wondering why the same-token keys would be suppressed, and whether that is about avoiding repeats or just a side effect.

I'm settling on shuffled text at about 0.7 and as-is at 0.3, which leaves Python the most likely corpus at around 0.5, English near 0.34, and romaji near 0.16.

I'm now assigning labels to the layer 0 heads: L0H0 reads as a broad-ish local window, and L0H1, L0H2, L0H4 and L0H5 are close to uniform attention that mostly pass along positional information. L0H3 looks like a previous-token or local head, which I'm still checking.

I'm settling on L1H0 through L1H5 as broad heads whose keys mostly come from the earlier attention outputs, with the query position ignored. For L0H3 I'll just call it a local head, since the shuffled-text story fits that better than a strict previous-token head.

I'm labeling the later layer heads: L1 and L2 heads read as broad, with L3H0, L3H1, L3H2 and L3H5 looking similarly broad. L3H3 is the open case, a content-similarity head with some induction-like cross-attention, and I'm leaning toward calling it a weak copy head.

I'm now settling on "unclear" for L3H3, since its dominant pattern is general content similarity and the induction-like cross term is weak, so calling it a copy head would overstate it. Then I'll move on to L3H4, which looks broad.

I'm going to write up what the L3H4 numbers show: its query-key term is small, it has no key bias, and its value-output term is the largest of the group, so I'll state those figures directly.

