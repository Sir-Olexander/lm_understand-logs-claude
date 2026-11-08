I'm looking at the eigenvalues, which come in complex pairs with large angles. That pattern points to a rotation or shift operator acting on a positional subspace, where the score peaks at a nonzero offset rather than at zero like a standard self-attention head would. A large negative real eigenvalue at 180° would suggest alternation.

The 180° eigenvalue, with its large negative real part, suggests the head is keying on odd offsets, most plausibly the previous token. The positional frequencies are consistent with a single sharp offset of 1. A position-embedding subspace is my assumption, though, so I want to check whether a token-embedding interpretation would fit the data better.

So this head is a previous-token head, and I'm wondering whether that actually makes sense for shuffled text. In a shuffled line, the immediately preceding character only tells me it belongs to the remaining multiset, so a sharp prev-token head seems nearly useless there, while the broad heads would carry more signal.

I'm now weighing whether an induction-style head at layer 1 or 2 could read the prev-token signal from L0H2, with keys carrying the current token identity. Before settling on that, I want to check the L0H3 numbers, since its more symmetric QK pattern doesn't obviously fit a sharp positional role.

Now I'm checking whether induction could be built from L0H2's prev-token signal, looking at how strongly the L1 keys pick up that subspace. The enrichment there is close to random, so no clear induction route through it. I'll test whether L1 outputs feed the later layers' keys instead.

The induction question is still open. L1's keys lean on the structural channel more than on prev-token information from L0H2, so I'm not seeing an induction route there. Wikipedia text and Python identifiers repeat often enough within 128 characters that induction would normally pay off, which makes its absence here worth noting. I'll check whether later layers read L1's outputs instead.

I'm checking whether v also lies in the positional subspace. If it does, the whole head depends only on positions, and its output would be a function of position alone, which would be a strange but possible role for a fixed positional bias in L1.

I'm wondering whether v also reads token content rather than position, which would make the head's output depend on what kinds of tokens it attends to. If keys favor something like newlines, the output would encode distance to those tokens, and combined with the position it would give something like position within the line. I'd then check whether the L1 keys pick that feature up.

I'm checking whether the key vectors of those four L0 heads overlap more with the positional subspace or with the token subspace from L0H2's output, since that tells me whether they're reading position or content.

