I'm finding that the attention heads each prefer different token sets, so there's no single newline preference. For the corpus inference, I'm leaning on a T-dimension around 24 to 26, one late induction head, purely positional layer-0 attention, and local layer-1 behavior. In Python, the column relative to the newline seems key, and I suspect layer 1 could compute it through token-preferring heads.

I'm picturing indentation runs as a counting problem: a local head could average "is space" over the positions since the last newline, but that needs the newline position to work. Pure reasoning isn't getting me further here, so I'm considering checking whether the "massive" outlier dimensions track position.

I'm weighing whether the top binary feature is "space vs letters" or "vowel vs consonant," since the token-set sizes would change how it shows up in the geometry. A single-token feature would give a dual vector that matches just one embedding, but without labels I can't check that. I'm going to try labeling tokens through the induction and previous-token head geometry instead.

I'm working through a congruence problem: the induction head's bilinear map should be diagonal in the token basis, but without the embedding matrix I can't recover that basis from the eigendecomposition alone. I'm considering whether pairing it with a copy map that is diagonal in a similarity sense could pin the token coordinates down.

I'm seeing that two bilinear forms that are both diagonal in the same unknown token basis could be simultaneously diagonalized, with the generalized eigenvectors recovering individual token directions. The first candidate for such a form is the induction matching map, where a query token matches a key token only when they are identical.

I'm checking the L1 "avoid same token" forms next; they look like a negative diagonal plus low-rank token preferences, so they aren't purely diagonal. Combining them with the induction form might still break the clean diagonalization I was hoping for.

I'm considering whether the prev-token head maps tokens to nearly orthogonal vectors, which would make its Gram matrix roughly diagonal in the token basis, though that isn't guaranteed. The induction match matrix should be fairly diagonal in token coordinates as well.

I'm considering whether the "same" form from L3H3 might also be roughly diagonal in the token basis, which would give a second candidate for simultaneous diagonalization alongside the induction match. Even if I can recover token directions this way, I'd still need to work out what each token means, since the geometry alone doesn't say which one is a space.

I'm weighing whether the count of strongly copied tokens says anything about the language: about 18 strong prev-token singular values could fit either letters plus space and newline, or the romaji split of vowels and consonants. Without labels, though, that count doesn't settle which one it is.

I'm counting how many tokens are frequent enough to show strong prev-token copying, and comparing English letter frequencies against romaji and Python. Each candidate gives a rough count of about 17 to 22 frequent tokens, so the count alone doesn't separate them.

I'm checking whether the uniform head L0H5's third value direction reads a token feature like "is space" or "is vowel," and whether that direction ranks among the most important token features in the first MLP layer.

I'm accepting that the T-dimension question may stay open, but I want to recheck whether LayerNorm centering changes the count of token directions. Subtracting the mean from each token embedding doesn't alter the span, though the normalization by sigma depends on the combined token and position norms, so the token component gets scaled per position. I'm also wondering whether any tokens in a romaji corpus would never appear and so have unused embedding rows.

I'm weighing whether the 26 robust token dimensions, with no weak tail, fit English, romaji, or Python better, since rare tokens like l, v, x, q would have shown up as weaker dimensions.

Ruling out Python: the neat 26-dimension count fits English best, since English would give about 24 robust letters plus newline and x/j, with q and z too weak to show. Romaji comes out a little low and Python a little high, and a single late induction head points the same way. I'm leaning toward English at about 0.5, with romaji still possible.

Weighing the remaining options, I'm settling on python at about 0.23 and romaji at about 0.27, so English stays the lead without any ties at the top. The unclear part is the L0H0 offset: I want to estimate it from the L0H3 R-value, which should be simpler than working it out directly.

I'm comparing the eigen-angles of L0H0 and L0H3 in the shared P-subspace, checking whether L0H0's are smaller so it attends nearer, including itself. Its top eigenvalues are real and positive, which points toward self or nearby attention. I'm also planning to solve for L0H0's bilinear form as a weighted sum of L0H3's shifted powers.

I'm going to test whether L0H0's bilinear form can be written as a combination of powers of L0H3's rotation, using the orthogonal polar factor of B3 restricted to the shared subspace. If that fits well, L0H0 is a positional mixer over nearby offsets. If the two forms don't commute, that assumption fails and I'd fall back to comparing eigen-angles directly.

I'm sorting the layer-2 heads by how far back they look: most seem local, with L2H3 looking broader. L2H4 is unclear, since its alignment with earlier heads flips sign depending on which one I compare against. Before I commit to labels, I want to recheck the sign of the coord-21 key bias.

Checking the sign of the coord-21 key bias: if the L0H5 averaged feature decreases with position, a positive bias would pull attention toward early tokens, which looks more like a sink than recency. I need to settle that before calling the L1 heads local.

I'm checking whether the L1 heads' behavior points to a recency pattern or to a fallback on the first position, since the positional key bias on coordinate 21 could go either way depending on its sign. I'll tentatively call the L1 heads local, with the positional signal and the token preferences together explaining their attention.

I'm sorting the heads by how they weight position. Layer 3 and layer 0's H1, H2, H4 look broadly spread, and L0H0 looks local. L0H3 is clearly the previous-token head, and I'm unsure whether the layer-0 spread heads might instead attend to the first position.

I'm going to test whether the layer-1 and layer-2 heads show recency through a rank-1 query-key pattern, by checking how the scores move with position along the slow positional direction. Without the positional vectors I can't confirm the monotone trend, so for now I'll just label those broad heads as averaging positional embeddings.

I'm going to verify the L0H0 offset directly: fit B0's positional form as a combination of powers of L0H3's polar rotation, using the top 16 positional directions.

I'm polishing the rotation fit: checking whether the orthogonal factor of B3 can reproduce the positional form of B0, and whether it fits well enough to call L0H0 a positional mixer.

