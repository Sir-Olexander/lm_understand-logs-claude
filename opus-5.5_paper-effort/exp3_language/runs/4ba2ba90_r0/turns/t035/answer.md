**Observations**

There are four main findings.

1. **Layer 0 splits into three kinds of head.**
   - **L0H2 is a sharp positional head.**
     - Its query-key matrix (Wqᵀ·Wk, with ln1_w folded in) has Frobenius norm 32.6. The other layer-0 heads range from 0.24 to 3.2.
     - It has about 24 significant singular values (11.3, 10.2, 9.7, 9.6, …).
     - The query and key sides span the same subspace (principal cosines about 0.99 over 20+ dimensions).
     - Its eigenvalues are complex pairs: 8.69 at ±52.6°, 7.84 at ±103.8°, 7.72 at ±68.9°, 6.85 at ±156.2°, 6.33 at ±32.3°, plus a real 6.80 at 180°.
     - This is a shift operator on positions, which favours a fixed small (odd) offset. The 180° term favours odd offsets.
     - Its value-output (OV) matrix reads a different, roughly 20-dimensional subspace with a flat spectrum (0.79, 0.72, 0.68, 0.67, …). That looks like copying the identity of the attended character.
   - **L0H0, L0H1, L0H4 and L0H5 attend almost uniformly.**
     - Their query-key norms are 0.44, 0.24, 0.85 and 0.62, and are essentially rank 1.
     - Their OV matrices are rank 2–3 (8.07/3.33/2.10, 5.66/3.83/1.07, 3.40/1.64/1.08, 8.33/3.61/1.68).
     - All four read the same direction, which sits in L0H2's main positional plane.
     - All four write almost only to residual coordinate 222 (loading −0.92; output overlaps 0.96–0.99 between heads).
   - **L0H3 is a small positional head.** It is mostly rank 1, in the same plane, with a positive real eigenvalue (2.09).

2. **The coordinate-222 channel is a control signal for later attention.**
   - ln2_w at coordinate 222 is about 0 in layers 0–2, so the MLPs ignore it. ln1_w.1[222] = 1.12 is the largest in that layer, and ln1_b is large there (−1.47, −1.38, −0.61).
   - Every layer-1 key matrix composes strongly with L0H0/1/4/5 (K-composition 0.50–0.57; random baseline 0.05).
   - About 55% of each layer-1 key matrix's energy lies in these 3 dimensions.
   - The key preference set by the query bias lies 87–90% in this channel.
   - Layer-2 keys (17–21%) and the keys of L3H2–H4 (9–11%) also read it.
   - Layer-1 queries read the current character and L0H2's output (18–25%). They are strongly negative on "same character" (trace/norm −1.7 to −2.7 on the character subspace, where the maximum is about 4.5).

3. **L3H1 is an induction head.**
   - Its query (current character) matches the key's previous character as written by L0H2: score 3.15 out of a maximum of about 4.5. No other head exceeds 0.75.
   - Its OV norm is 40.9. Its spectrum has 24 strong values (down to 5.2), then 5 weaker ones (4.3 to 3.3).
   - It writes into all 27 output-character directions (singular values 7.5 down to 2.6).

4. **The output side shows that all 28 characters occur, 5 of them rarely.**
   - W_proj.3 has one sparse "massive dimension" direction (26.3, mostly coordinate 103).
   - It then has 27 dense directions: 22 strong (20.7 down to 10.4) and 5 medium (8.35, 8.06, 7.91, 7.46, 7.17). After that come sparse directions (5.8, 4.1) and a tail at or below 3.2.
   - 27 is 28 characters minus one, so every character is a live prediction target.
   - The copying head's 5 weaker directions map onto exactly the 5 medium output directions (overlaps 0.52–0.65). So those 5 rare characters also occur in the input and get copied.
   - The near-uniform head L3H5 (query-key norm 6.68, OV norm 45.4) boosts mainly one of the 5 rare directions (gain 9.1, against about 2–3 for the others). That looks like a rare character predicted from broad context, such as the newline.

**Heads**
- L0H0: broad — rank-1, tiny query-key matrix (0.44); averages one key-selected feature; writes a scalar to coordinate 222.
- L0H1: broad — same pattern (query-key 0.24); same input and output directions as L0H0.
- L0H2: previous — large, high-rank positional shift (complex eigenvalue pairs, 180° term favouring odd offsets); about 20-dimensional OV that copies the character.
- L0H3: local — small, mostly rank-1, symmetric positional query-key matrix (eigenvalue 2.09 at 0°); rank-5 OV.
- L0H4: broad — rank-2 query-key matrix of norm 0.85; writes coordinate 222.
- L0H5: broad — rank-1 query-key matrix of norm 0.62; writes coordinate 222.
- L1H0–L1H5 (all six): local — keys about 55% in the coordinate-222 channel (likely a segment or word-boundary signal); queries read current and previous character; avoid same-character keys (−1.7 to −2.7); small positional component.
- L2H0: local — keys about 20% in the coordinate-222 channel; avoids same character; weak evidence.
- L2H1: unclear — the query's previous character matches the key's character (1.36); purpose not identified.
- L2H2: local — same pattern as L2H0.
- L2H3: local — same pattern as L2H0.
- L2H4: local — same pattern as L2H0.
- L2H5: local — same pattern as L2H0.
- L3H0: unclear — content matching (trace/norm +1.53, previous-to-current +1.35); writes little to the output.
- L3H1: copy — induction match 3.15; OV covers all 27 output directions.
- L3H2: unclear — keys read the coordinate-222 channel (17×); writes the rare direction.
- L3H3: unclear — similar to L3H2.
- L3H4: unclear — similar to L3H2.
- L3H5: broad — weakest query-key matrix (6.68) with the largest OV (45.4); boosts mainly one rare output direction.

**Reasoning**

*Order.* A sharp head that copies the previous character, and an induction head, are only useful when characters keep their order. In shuffled lines, every earlier character of the line is equally informative and repeated text cannot be copied. A shuffled model should be almost all near-uniform heads. So the text was used as is.

*Corpus.* The output side has 27 live directions, so all 28 characters are used. Exactly 5 of them are rare but present (they are copied).
- **English fits best.** It has 23 frequent characters (space plus every letter except x, j, q, z), which gives 22 strong directions. It has 5 rare ones (x, j, q, z, and newline, since Wikipedia lines are long paragraphs). A rare newline predicted from broad context matches L3H5.
- **Python fits less well.** Newline, x, k, v and w are common in code; only q, z and j are rare. I would expect about 24 strong and 3 rare directions. Python's heavy repetition should also produce earlier and stronger induction heads than one in the last layer.
- **Romaji fits worst.** l, q, v and x essentially never occur in Hepburn romanization, so those output directions could not be trained and could not be copied.

corpus: english
order: as is
probabilities: python=0.20 romaji=0.10 english=0.70; as is=0.92 shuffled=0.08