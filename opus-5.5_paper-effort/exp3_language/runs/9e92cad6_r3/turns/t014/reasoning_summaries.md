I'm checking how much of the norm a random subspace of each size would capture, which gives 0.22 for 18 dimensions as the baseline to compare against. Against that, the L1 query vectors read H3o at 0.35 to 0.43 and T at about 0.3, while the key vectors pull strongly from Fout at 0.63 to 0.68, well above its 0.07 baseline.

Now I'm working through the L1 heads, which all look nearly identical in structure, and wondering whether the value reads of H3o point to a previous-token or a local-window mechanism.

I'm working out what a prev-token pattern would require in the position subspace: the rotation angles of the QK block would have to match the frequencies of the position embeddings. The measured angles all sit near 45°, which would mean every frequency has roughly period 8. That seems unlikely if the attention is meant to be sharp, and the eigenvalues alone don't reveal the actual frequencies.

Now I'm weighing whether the sharpness of L0H3's scores, with the top singular value near 1.95 and the position components of norm around 10, would be enough for a sharp prev-token pattern. Then I'm turning to L0H0, whose position-subspace QK has real positive eigenvalues, which would suggest attention to self or nearby tokens instead.

Now I'm wondering whether Fin, the input read by the L0H1, H2, H4 and H5 OV circuits, sits inside the positional subspace, which would make those heads output the position of whatever they attend to. Their rank-1 QK directions draw only moderately on T and P, so the token-embedding slice I'm using may be missing something, possibly rare tokens or whitespace that L0H3 handles specially.

I'm considering that the L1 heads might be attending within the current line, reading token identity to build up a bag of characters. The query-independent pull toward Fout keys would fit a mechanism where L0 locates the last newline and the L1 keys carry distance-to-newline information.

I'm asking whether the L1 heads' value reads of H3o fit a shuffled-text setting, where previous-character adjacency carries little meaning. If L0H3 instead attends to itself and copies token identity, then the L1 value reads would be carrying something other than neighbor information.

I'm weighing whether L0H3 is a self-attending head that just reshapes token identity, which would fit a shuffled-text setup, versus a previous-token head that would fit the classic as-is circuit with L3H3 as an induction head. I need some test that separates those two possibilities.

I'm checking whether L0H3 is attending to itself or to a previous position. A self-attending head would have a symmetric positional score matrix with real positive eigenvalues, but the complex eigenvalues at roughly ±45° and the strong asymmetry suggest it attends at a nonzero offset. That makes L0H3 look more like a previous or local-window head than a self head. I'm also wondering whether a shuffled-text model would ever need a local positional head like that.

I'm setting up a test to tell an induction head apart from a same-token matcher: compare the L3H3 score for query-side current-token against key-side previous-token reads with a baseline using raw token identity on both sides. The earlier score of 0.96 at k=10 is much higher than the other heads, so I want that baseline before drawing a conclusion.

I'm setting up three candidate matrices to test which pairing of query and key reads fits the induction pattern: current token on the query side, previous token via L0H3 on the key side, plus same-token and previous-token-matching variants. I'm also planning to run the same positivity check on the L1 and L2 outputs as possible carriers of previous-token information.

