I'm checking the singular values for L0H3's OV matrix: about 18 stand out well above the rest, then they taper off gradually. That suggests roughly 18 to 20 strong dimensions.

I'm weighing whether the strong singular values really mean something about the token set. Romaji and English both land somewhere around 22 to 24 usable symbols, which is close to the roughly 18 to 27 dimensions the OV might need, so the gap doesn't settle it. I'm also thinking about a different angle: Python indentation creates runs of leading spaces, which could make counting-like behavior visible in a head.

I'm turning to the Japanese romaji model, where local context matters and copying helps less. Then I'm looking at English text, where predicting the current word needs about 5 to 10 characters of local context, so several local heads would be needed. The concern is that the single local head plus the layer-1 heads have no way to tell which token is most recent, since position isn't read by the queries and keys, which makes order tracking beyond that window very hard for the model.

The next step is to check whether later-layer heads can read position at all. If their queries and keys ignore the positional subspace, their attention can't depend on distance, and the only positional signal they get is an absolute, query-independent bias on each key.

I'm thinking about whether later-layer heads can even track which line they're in, since their queries and keys seem to ignore position. Without that, the shuffled model would have to rely on content similarity or the broad local windows from layer 0 to approximate the current line, which matters more for short lines than for long paragraphs.

I'm now noting that in a shuffled English or romaji model, the window is mostly one line, so a bag of the whole window roughly equals the line, and the local heads cover the rest without needing positional attention in later layers. Shuffled Python is different, since its short lines mean line tracking would have to come from positional or newline mechanisms, and as-is text would need relative positional attention that this model doesn't seem to have.

I'm now looking at the L3H3 similarity head, which in the shuffled case would attend to positions with similar local bags, and at a possible "avoid my own token" head in L1. I'm wondering if it's really a without-replacement correction, but that doesn't fit well with long lines where the counts are large.

I'm now checking whether the shuffled-English-or-romaji reading holds, and whether later layers really ignore position. The positional subspace was estimated from L0H3's top directions, and L1's keys read it about twice as strongly as the other heads, so I'm wondering whether that overlaps with the output space and muddies the conclusion.

I'm now reasoning that later-layer heads can't read the positional subspace in their queries and keys, so the only position signal they get is the absolute, query-independent bias on each key. That means they can't do previous-token or local attention except through content similarity, and I'm checking whether that's plausible for an as-is model.

I'm wondering whether the induction-style reading holds here, since it would mean later heads don't need position at all. Checking the L1 query-token and key-token blocks next.

Pure spreadsheet-level detail aside, the next stretch is weighing whether the as-is model is plausible at all: later heads show no induction pattern and seem blind to position, and with only one clearly local head that looks implausible for training on this much text, since local order carries a lot of predictive value. That points toward a shuffled setup. I'm also questioning whether L0H3's peaked, recency-focused pattern makes sense in shuffled text, where long lines would make a broad window more useful.

Next I'm looking at whether the line boundaries help: after a newline, the characters that follow come from a fresh paragraph, so a short local window would adapt to that quickly. L0H3 might also just be a self-plus-recent-characters head, giving a small local bag, and I'm checking how much MLP0 leans on its output.

I'm weighing whether an as-is romaji model could work: a single previous-token head would cover bigrams, but trigrams would need later-layer attention to target the position before the last one, and those heads seem to ignore position. Content-based attention by itself doesn't pick out that exact earlier slot, so this reading looks unlikely.

I'm checking whether L0H3's pattern is really sharp or just a broad recent window, since the rotation angles suggest low-frequency planes. In a shuffled model, a wide window like that would be useful for approximating the current line's bag of characters, and L0H0's symmetric head might cover the same role.

I'm now checking whether any later-layer heads spread attention almost uniformly, since L3H4 looks like a whole-window bag with a large output projection. If that holds, the model gets its order-free summaries from several window sizes at once.

I'm weighing whether the broad whole-window bag head in layer 3 matters most, since its output norm is the largest in the model, which points toward a shuffled setup. Checking whether the shuffled Python case, with its short lines, would need line-specific bags and indentation signals.

I'm working out what a line-tracking mechanism would need in this architecture: a key-only bias that stays constant within the current line and drops for earlier lines. The strongest content-weighted head, L0H1, might attend to newline tokens and so pick up line structure, which would complicate the position-blind reading.

I'm checking whether L0H5's larger output magnitude, compared with L0H1, L0H2, and L0H4, means its running mean dominates the per-unit contributions to the layer-1 key bias.

I'm checking whether L0H5's rank-1 QK term, though small, weights recent positions enough to matter, since its output gain is so large that even mild weighting would shape the key bias.

I'm now weighing whether the English-versus-romaji difference matters for the line-level machinery: if lines are rare within the window, the extra heads would mostly act as a sink, but if lines are frequent, as in Python, they would track line structure. I'm also checking whether the letter distributions differ in how peaked they are, since romaji's vowel-heavy frequencies would look very different from English's flatter spread.

I'm now weighing whether romaji's letter frequencies stay stable across texts, which would leave the window bag with less to add. I'm also asking which features in the weights might track how many distinct symbols are active, since the strong OV dimensions in L0H3 could reflect that count.

I'm looking at the rest of L0H3's singular values: they keep tapering gradually past the strong cluster, with no sharp cutoff near 24 or 27. The tail values are still well above the random-init noise floor, so they look learned rather than incidental. I'm also starting to check whether the token embedding's centered rank tells me anything about the active symbol count.

I'm going to estimate the token subspace by stacking the normalized value matrices of L0H3 and L0H0 with the normalized first MLP weights, then checking whether the spectrum shows a clean gap. I expect it won't, so I'll also try counting the rarely used romaji tokens, whose embeddings stay close to initialization.

I'm comparing the spectrum of L0H3's V matrix against its OV matrix: V's learned values sit only slightly above a random-init floor around 0.24, while OV's learned values stay well above its much lower floor even deep into the tail.

I'm weighing whether shuffled text would push the model toward line-tracking, since that depends on how much line structure matters in English or romaji compared with Python. I'm checking whether the outs mechanism could be doing that work, though the tail of the singular values alone doesn't settle it.

I'm checking L3H4 next: its OV is the largest in the model, and its QK is nearly flat, so it spreads attention across the whole window. In English or romaji, with long lines, that window mostly covers the current line, which makes it a strong statistic under shuffling. In Python, with shorter lines, it would mix several lines and be less useful.

Still weighing whether line-level tracking would be needed under shuffling, especially for Python where newlines are frequent and for English or romaji where lines run long and the window mostly covers the current line.

I'm considering whether the model might distinguish English from romaji through its token subspace, perhaps via heads that separate vowels from consonants, since vowels make up a larger share of romaji letters. I'm also wondering if the L1 anti-duplicate behavior would look different across corpora, though I don't have a basis to decide that yet. I'm now turning to a different angle: checking whether the norms of token embeddings track frequency, since rarely used tokens should stay closer to initialization.

I'm considering whether the token embeddings are roughly orthogonal, which could show up as separate, identifiable directions in the input weights of MLP0 that respond to specific current tokens.

I'm thinking about clustering the rows of W_fc.0 after projecting them onto the token subspace, on the idea that single-token detector neurons would point toward specific token embedding directions and give roughly as many clusters as active tokens. It's a long shot, so I'll first need a reasonable estimate of that token subspace, since L0H3's V alone may not span all the token dimensions.

I'm going to take L0H3's value input space, roughly the top 25 dimensions, as the token subspace and project the MLP0 weight rows onto it. Then I'll normalize them and look at how they're distributed, possibly with k-means or by sorting on norm.

