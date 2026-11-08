I'm checking the cosine values for the first matrix at 20 positions, and there's a sharp drop after the first 27 dimensions, from around 0.45 down to 0.23. At 40 positions the drop is in the same place, so the shared token subspace looks like 27 dimensions.

I'm weighing whether the sharp drop at 27 shared dimensions could partly reflect positional components rather than token identity, though the knee stays put across sequence lengths, so I won't dwell on that. Romaji looks unlikely at around 0.1. Now I'm trying to tell English from Python, and the newline and indentation-space statistics aren't decisive yet.

I'm checking whether the L1 heads could count spaces in an indentation run. Their values read token identity rather than position, so they don't carry a position signal, and an anti-self head at a space would skip spaces and land on the last non-space token. That suggests the counting mechanism isn't in L1 at all.

I'm now weighing whether the L1 anti-self heads could help with English versus Python, but on reflection, avoiding self-attention is generic and doesn't clearly separate the two. I'm also considering whether L0H4's uniform output encodes position relative to a window start, though that looks generic too. I'd like to label the tokens directly, perhaps through the copy and induction circuits, to settle it.

I'm weighing whether the copy heads (L2H1, L3H1, L3H5) point toward English or Python, since a 4-layer model on English text would typically have one to three induction heads, while Python would likely show more. I'm not settled on which way that cuts yet.

I'm looking at L3H3 next: its QK pattern is fairly flat and its output is strong, so it's a broad averaging head. Its value input looks like it's pulling from earlier layers rather than from position or raw token identity, so I'd need to trace what those features are.

I'm now looking at the layer-2 heads with large positional key biases, L2H2, L2H4, L2H3 and L2H0, to see whether they act as recency or sink heads. I'm also thinking about whether a Python-specific mechanism could attend to the start of the current and previous lines, which would need a newline feature in the keys, and I'm checking which of L1H1, L1H5 and L2H4 might carry that signal.

I'm checking whether the L1 and L2 heads share a common preferred key token, which would point to a landmark token like newline or space. I'll compare the token-projected key-preference vectors across those heads and against the key-token directions of L0H0 and L0H2.

