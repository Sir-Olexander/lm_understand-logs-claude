I'm reading the L3H5 scores, where the o5 and o1 matches are strongest. It looks like a same-context match, with the query's previous token lining up against the key's previous token, and I'm checking whether the current token T also lines up with the key's previous token.

I'm weighing whether the T-to-o5 match is the induction pattern, with the query's current token lining up against the key's previous token. The o5-to-o5 and o1-to-o1 scores seem to point the other way, toward matching local context rather than induction, so the two readings don't yet agree.

I'm checking whether the diagonal matches (o5/o5 and o1/o1) fit induction, since induction would need the query's current token to line up with the key's previous token, which looks more like an off-diagonal pattern. The symmetric diagonal scores instead suggest the head matches the same offset features, which doesn't look like induction, unless the head attends to positions where the previous token matches and then reads the token that follows.

I'm working out what L3H5 actually computes: its strongest match is same-local-context, not induction, since the o5 and o1 diagonals dominate. Its value side reads the current token, so the head seems to pick up positions with similar local composition rather than the token that follows a repeated pattern. I'm now turning to L2H1, where the query's current token against the key's o5 looks like the induction-style alignment I was hoping to find.

I'm checking L2H1 and L3H1 next, where the query's current token lines up against the key's o5 feature, which is the induction-style alignment I was looking for. The L2H2 scores point the other way, with the query's previous token matching the key's current token, so that head looks like it searches for earlier occurrences of my previous token instead.

I'm weighing whether o5 (L0H5) and o1 (L0H1) are sharp single-offset heads or bag-of-local-context heads. Their outputs sit in mostly distinct subspaces (max cosine 0.35), which fits either reading, so I'll check how the current token T maps through their OV circuits to see which one holds.

I'm trying to pin down whether the o5 and o1 features behave as sharp single-offset heads or as broader local-context bags. The distinct output subspaces fit either case. I'm also weighing the L1 heads' QK, where o5 matches at both query and key, to judge how much of the attention comes from positional signal versus token content.

I'm working through whether the positional signal in the LayerNorm weights is axis-aligned, since that could tell me where the position embeddings sit in the residual stream.

I'm looking at the eigen-angle spread for L0H5 on its own: the angles run from near 0 up to about 110 degrees, which is more than a narrow prev-token head would need. I'm weighing whether it's a broad local-window head instead, where the Fourier response concentrates at low frequencies.

I'm treating the 20-odd significant QK dimensions of L0H5 as evidence of a sharp kernel, since the magnitudes stay fairly flat across many frequencies, which fits a previous-token head more than a broad window. M1 by contrast has only a few significant components, which would suggest a narrower, more low-frequency pattern.

I'm checking whether M5's spread of eigen-angles, which run from about 7° up to 180°, means it's a sharp, previous-token-style kernel rather than a broad window, since a window head would only use the low frequencies.

I'm checking whether the strong plane shared by M5 and M1 is a slowly varying, absolute-position component, where neither offset hypothesis would fit cleanly. The high-frequency angles look consistent with M5 at offset one and M1 at offset two, apart from some drift of 15 to 30 degrees, so the strongest plane may be what breaks that picture.

I'm trying to pin down what the L0H4 averaging plane is doing. If it's slowly varying, it could be carrying a broad, far-back signal, and both H4 and H0 read from it. The amplified OV singular value of 13.88 on L0H4 suggests that plane matters, so I'll look for a more direct way to get the frequencies.

I'm considering whether the positional planes could be read through MLP0's neurons, since a neuron reading the slow positional plane might act as a position threshold, though that wouldn't pin down the frequencies. Another route is to fit each head's positional QK eigenvalues jointly across L0H0, H2, and H3 to recover Fourier coefficients.

I'm now weighing a fit for the fitted-frequency idea against a simple consistency check: each head's positional eigenvalues should look like a single offset kernel. Fitting ten frequencies jointly across five heads is too many unknowns to be worth it, since several planes carry tiny magnitudes. Given the two strongly positional L0 heads with token-copying OV circuits, I'm inclined to keep the offset-head reading as the working picture for now.

I'm checking the asymmetry in the o5 and o1 scores across L2H1, L3H5, and L2H2, to see whether the pairing points toward one offset assignment or the other under the induction alignment.

I'm checking the asymmetry in the o5 and o1 scores: the T-to-o5 and o5-to-o1 pairings are elevated in L2H1, L3H5, and L3H1, while the reversed pairings are lower. That pattern looks like the induction signature, which would mean o5 is the previous-token offset and o1 sits one step further back.

I'm weighing whether the same-offset matches on the o5 and o1 diagonal, which are strong in L3H5, mean the features aren't perfectly single-offset and each partly attends to its neighbors. That mixing would blur the clean induction picture I've been building, and it's the kind of detail I'd need to check before trusting the offset assignments.

Checking whether the asymmetry in the o5 and o1 pairings holds up under exact block norms compared with the reversed pairings. I'm also going to look at whether the V vectors of L2H1 line up with that pattern.

I'm checking the OV trace on L3H5, which is negative at -0.39, and wondering whether that conflicts with copying. But the output directions don't have to line up with T for an untied unembedding, so a negative trace in the residual basis doesn't rule out copying on its own. I'll verify the o5/o1 asymmetry using finer subspace definitions.

I'm checking whether o5 or o1 is the prev-token offset, comparing T→o1 and o1→o5 against the o5-based pairings. Those alternatives come out low, so the data supports o5 as offset 1 and o1 as offset 2. The subspaces also stay distinct enough, with overlap at most 0.35, and L1's T/T score of 3 is a separate signal I'll note.

Layer 1 looks different from L2 and L3: its strongest cross-offset pairing runs the other way, with the query's o5 matching keys that carry the current token. That would mean attending to positions whose current token matches my previous token, and I'm trying to work out what that lookup would pull forward.

Now I'm checking the asymmetry in the o5 and o1 pairings more carefully by computing precise block norms and comparing them to the reversed pairings. I'll bootstrap across subspace dimensions (8, 12, 16) to see whether the induction-style pattern holds up.

