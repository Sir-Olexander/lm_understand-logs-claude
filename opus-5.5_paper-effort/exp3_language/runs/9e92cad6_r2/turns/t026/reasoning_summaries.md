I'm looking at the singular values for this head's OV matrix: eight dominant ones standing out, then a gradual falloff. That suggests the broad bag head writes around eight main directions strongly, and those could be the principal components of letter-frequency variation, like topic or style.

I'm weighing which shuffled corpus would produce a strong local head like L0H3, and I'm reasoning that under shuffling the local order is mostly noise, so the gradient wouldn't build a sharp positional head. The as-is corpus looks like the case where that head is essential, especially for romaji, where the next character depends on the previous few.

I'm checking whether the L1 heads' self-avoidance, through both their token and their previous-token matching, could implement "attend to the most recent other position," which would give the as-is model a way to build trigram context. That mechanism would matter less in shuffled text, where excluding self mainly reduces redundancy rather than being essential.

I'm weighing whether the key bias is recency-based or a broad sink, since that determines whether L1 heads act as local "previous token" readers. The running mean from L0H5 is the odd part: a plain positional feature would already serve as a recency signal, so I'm asking what a content-weighted running count would add that raw position can't.

I'm checking whether ins is really positional by comparing it against the token subspace, where the overlap is tiny, and it sits inside L0H3's key subspace, which fits position. The MLPs and later value reads ignore it, so it's either a positional signal or a constant. I'm wondering if it specifically marks the first position, since learned embeddings often make position 0 stand out, and then a uniform running mean of that indicator would decay as 1/(j+1).

I'm checking whether the 2D subspace could mark early positions in general, since outputs scaled by 1/(j+1) would give a sink-like pattern. Testing that against L0H3's QK is tricky: its strongest key directions lie in the ins subspace, and a previous-token head relying heavily on a position-0 feature wouldn't make much sense.

I'm treating the dominant positional plane as the source of the running mean, which would trace an arc over the sequence. A key bias built from that mean could favor either recent or early positions, and since L0H5's output gain is large, the resulting swing across positions could be very sharp, pushing the L1 heads toward strongly position-driven attention.

I'm considering two outcomes for the L1 heads: with strong recency and self-avoidance they'd act like previous-token heads, a local design like the as-is model. With strong primacy they'd sit idle on position 0 unless content overrides, leaving content-matching heads that aggregate specific letters.

I'm weighing whether the running-mean mechanism is a deliberate "soft sink," where the feature decays as 1/(j+1) over the first several positions, instead of the usual primacy built directly into the position-0 embedding.

I'm considering whether the L0 group might track line structure rather than acting as a sink: a newline-preference logit that scales with query position could keep attention sharp on newlines as the context grows, so the output would encode the position of the most recent line start.

I'm reading the head's behavior as a newline detector: its content weighting grows with query position, and averaging over the positions of newline tokens would mark where each line starts. That would let later keys bias attention toward the current line.

I'm now asking whether this value direction marks spaces, which would make the running mean track word boundaries and let a head attend to the current word in English, or whether it marks newlines, which would fit a line-tracking role better, especially for shuffled text where spaces carry no real word structure.

Settling between two candidate stories now: one where the group tracks newline positions to follow line structure in shuffled Python, and one in which spaces, not newlines, drive the tracking as in as-is English. The shuffled-Python story hangs together better, since it explains the line-level aggregation, the self-avoidance, and the suppression from the negative OV values.

I'm weighing the two candidate stories against each other now, with the as-is English or romaji case on one side and the shuffled Python case on the other. The newline-tracking group looks odd for the English one, since newlines rarely fall inside the window there.

I'm now considering whether the key bias tracks where the current line begins, which would make L1 and L2 pull from the tokens on the present line. Under shuffled Python that would fit the strong bias magnitudes, since the heads would be aggregating line-local letters.

I'm weighing whether the group's content weighting should be strong for a frequent special token like space, which appears everywhere, or for a rarer one like newline that only matters in Python. I'm also considering whether the L0H3 copy behavior could reveal the same token identity, though that seems unlikely to settle it, so I'm leaning toward looking for a different test.

Comparing the two stories again: as-is English or romaji would show strong induction from all the repeated names and words, and this model shows little of that. Shuffled Python fits better, since long lines leave little need for heavy positional machinery.

I'm reconsidering the shuffled-Python reading: a short local window of recent characters could stand in for the current line, and the newline signal might mark where each line begins. Since induction heads usually form on natural text with plenty of repeats, their absence points toward shuffled data rather than as-is English.

I'm checking whether the outs bias reflects recency within the line or a general sink, and I'm looking at L0H3's query-key geometry to see if it attends to the recent past, which would tell me which one it is.

I'm checking the sign of the antisymmetric part of L0H3's query-key matrix, which rotates the logit by about -52 degrees, so the highest scores fall on past positions whose angle is 52 degrees ahead of the query's. That makes the positions drift clockwise as they get older, and I'm about to see whether the L1 key bias follows the same rotation.

I'm working out the L1 key bias for L1H0 from the L0H5 averaged input, using the per-head row (−48, 19) to weight each position's coordinates.

I'm going to recompute the query-key matrix and the bias weight in one shared basis, since the earlier matrix came from a different SVD. Once both sit in the same coordinates, I can check whether the bias grows or shrinks as the position index increases.

I'm still working through the head's query-key geometry: the symmetric part looks like a rank-one positive form, with the antisymmetric part carrying the rotation. Without knowing where the arc of positions sits, I can't tell whether the bias rises or falls, so I'm checking how L0H0's matrix lines up with the same ins axes.

I'm now checking whether the symmetric part of L0H0's query-key form, which is nearly rank-one, makes its attention favor recent keys for queries in one half of the sequence and early keys for the other.

Now I'm checking how the L1 heads' queries depend on local context: the query reads the token and the spaces at 1.5 to 2.4 times baseline, and the cross block between query-space and key-token is strong, so the attention preference depends on what the recent characters are.

I'm now checking whether the queries' dependence on local context points toward as-is text, since content-driven queries would fit natural language better than shuffled data, where the query's own bag should carry little attention-relevant signal.

I'm trying to pin down L0H3's attention kernel without the actual positional embeddings. A translation-invariant form would make the logit depend only on the offset, but with 128 unknown position vectors that's badly underdetermined. I'm considering whether the learned positional structure might just be smooth from shared attention patterns, though that's a loose argument I'm not ready to rely on.

I'm now weighing corpus choice by how often newlines show up in a 128-token window: Wikipedia text has few, Aozora romaji has about one, and Python has several per window.

I'm checking the sign of L0H1's negative top eigenvalue in its query-key form, which suggests a positional preference mixed with a token preference, though I can't yet tell which.

I'm settling on a per-head classification for the answer: L0H0, L0H1, L0H2, L0H4 and L0H5 look broad and near-uniform, L0H3 looks previous-token or local, and the L1 and L2 heads read as broad or local. For L3H3 I'm weighing whether its cross-structure reflects induction or plain co-occurrence, since those two computations look the same when the window is narrow.

I'm concluding the L1 heads look too homogeneous to be doing as-is induction, which points toward shuffled data at roughly 70-75%, and I'm about to work out which shuffled corpus fits.

Comparing the romaji and English options, I'm weighing how many lines each corpus would produce per window and how much line tracking would matter, with shuffled Python fitting that best. The count of strong token dimensions, roughly 18 to 20, is ambiguous: it could fit romaji better, but I still need to check whether the local-head copy dimensions only encode the distinctions that matter.

I'm checking the token-dimension structure of the first layer by projecting out the positional and attention-output directions, then looking at what's left of the token read space.

I'm going to project fc0 onto the leftover space, after removing the positional, S3 and S0 attention-output directions and the outs part, and then look at its singular values for a clear gap.

