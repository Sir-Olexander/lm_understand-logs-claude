**Observations**

- **Outlier dimensions.** Residual coordinates 21, 39, 58, 99 and 184 carry very large values. In `ln1_b.1` they are −0.43, −1.57, +1.37, −1.34 and −0.56.
  - L0H5 writes almost only into them: its OV singular values are 16.3, 8.8, 2.8, and the row for coord 21 has norm 14.7.
  - L0H1, H2 and H4 write a 2-D signal into coords 39, 58 and 99.
  - Every L1 and L2 head reads these coordinates as its two dominant key directions. Stacking the top-2 key directions of the six heads gives singular values 2.43 and 2.31, which is √6, so the direction is the same for all six.
  - The constant query (from the LN bias) puts a key bias of +16 to +29 on coord 21 in L1, +8 to +18 in L2, and only +1 to +4 in L3.
- **Layer-0 attention is purely positional.** I split the input into a token subspace T and a position subspace P using the layer-0 readers. The token-to-token part of each L0 head's QK has norm 0.02–0.15, against 1.2–2.1 in L1.
- **L0H3 is the strongest L0 head.** QK Frobenius norm is 14.1, with a flat spectrum (sv10 = 2.63).
  - Almost all of its QK eigenvalue mass is in complex-conjugate pairs at ±35–60°: 23.5 of 25.
  - Q and K read the same positional subspace (principal cosines 0.97, 0.96, 0.96, …).
  - V reads only the token subspace; its overlap with Q/K is at the random level.
  - So it selects by position and copies token identity, i.e. a previous-token head.
- **L0H5 attends uniformly.** Its QK norm is 2.6, below the random-init level of 3.1.
- **L1 heads avoid their own character.** The token-to-token QK form has a trace of −3 to −5 and is almost entirely negative.
- **L3H3 is an induction head.** I built the form "query = current token × key = previous token (written by L0H3)". For L3H3 its trace is 3.17, with 3.18 of 3.50 eigenvalue mass real-positive.
  - It also matches through L1H0 (1.03), L1H5 (0.84), L2H1 (1.08) and L2H5 (0.86), i.e. on longer context.
  - Its OV writes into W_proj.3's token-logit directions 2–24 with enrichment 2–8×.
  - No L1 or L2 head shows this circuit; the best is L2H4, weak and mixed.
- **About 26 token dimensions.** I removed the layer-0 attention-output subspaces and the outlier coordinates, then looked at what MLP0 (W_fc.0) reads.
  - Exactly 26 directions are token-subspace-enriched (3.3–7.7×, singular values 13.1 down to 4.9).
  - All 26 are non-positional: enrichment against all layer-0 Q/K readers is 0.2–0.6.
  - After them the token signal drops sharply (index 26: enrichment 1.0, singular value 4.3; index 27: 0.3).
  - This holds for different projection sizes.
- **Output side.** W_proj.3 has 21 large singular values (28.5 to 11.1), then 9.1, 8.4, 8.2, 6.4, 5.8, 4.6, 4.6, then 3.8.

**Heads**
- L0H0: local — positional QK (key directions 0.93/0.86 in P) with real-positive plus complex eigenvalues (near-self and short offsets); V reads tokens.
- L0H1: broad — rank-1, mostly positional QK (key ~0.3 in token space); V reads positions and writes the outlier code (39/58/99), i.e. a smoothed position summary.
- L0H2: broad — same mechanism as H1 (query directions cos 0.91 with H1).
- L0H3: previous — rotation-like positional QK (±35–60° pairs), Q and K in the same P subspace, V copies tokens; its output is the key for the L3H3 induction match (3.17).
- L0H4: broad — weaker copy of the H1/H2 mechanism.
- L0H5: broad — QK ≈ 0, so uniform; huge OV writes the coord-21 position code that later keys read.
- L1H0: local — constant-query key bias +29 on the position code; avoids same-character keys; Q and V read L0H3's previous-token output.
- L1H1: local — same pattern (bias +27, same-token trace −4.5).
- L1H2: local — same pattern (bias +22).
- L1H3: local — same pattern (bias +25).
- L1H4: local — same pattern, a bit weaker (bias +16).
- L1H5: local — same pattern (bias +29, same-token trace −5.2).
- L2H0: local — position-code key bias +15, weaker than L1 so broader.
- L2H1: local — bias +13; queries read L1 outputs.
- L2H2: local — bias +11.
- L2H3: broad — weakest L2 position bias (+6).
- L2H4: unclear — local bias +17 plus a weak, mixed-sign induction-like match (0.47 with L0H3, 0.56/0.64 with L1H1/L1H5, negative with L1H2/H3).
- L2H5: local — bias +15.
- L3H0: broad — small position bias (+3.3); positive symmetric content matching (trace 17).
- L3H1: broad — bias +1.5, diffuse QK.
- L3H2: broad — bias +3.1, diffuse QK.
- L3H3: copy — clear induction head: positive previous-token matching via L0H3 and L1/L2 heads; OV writes token logits.
- L3H4: broad — weakest QK of all (Frobenius 8.6, near-uniform); large OV into shared non-token output directions.
- L3H5: broad — bias +3.3; mild positive matching (trace 11.5).

**Reasoning**

*Order.* The text was used as is.
- A dedicated previous-character head (L0H3) and a clean induction circuit built on it (L3H3) are only useful if character order is preserved.
- Several local heads in L1/L2 point the same way.
- On shuffled lines, order inside a line is random, so the model would instead need broad "bag of the current line" attention, and the previous-token and induction circuits would be useless.

*Corpus.*
- **Romaji is unlikely.** MLP0 reads about 26 distinct, non-positional token dimensions. Hepburn romaji uses only 22 letters plus space and newline, so at most about 24 token directions; that is too few.
- **English vs Python.** English and Python both use all 28 symbols.
  - English fits slightly better: its rarest letters (q, z) may stay weak, which would leave about 26 strong directions. Python uses every letter fairly often, which would predict 27–28.
  - The architecture also fits natural language better: layer 0 is purely positional, layer 1 is six local "context excluding the current character" heads, and there is only one induction head, in the last layer.
  - On code, which repeats identifiers and indentation heavily, I would expect copying to appear earlier and in more heads.
- This corpus evidence is indirect, so the corpus is far less certain than the order.

corpus: english
order: as is
probabilities: python=0.24 romaji=0.21 english=0.55; as is=0.92 shuffled=0.08