**Observations**

- **Layer 0 splits into one sharp head and four weak ones.**
  - **L0H2** has a QK Frobenius norm of 32.6. Its eigenvalues come in conjugate pairs with large phases (|λ| 8.25, 7.7, 7.49, 6.78 at ±0.29π, ±0.58π, ±0.38π, ±0.87π, plus a real −6.53). Its query side and key side span the same ~23-dimensional subspace (cosines 0.99…0.9). That is the pattern of relative-position attention on a Fourier-like positional subspace.
  - L0H2's value/output (OV) map is high-rank and flat (singular values 0.79, 0.72, 0.68 … 0.3 at the 20th). It reads a different subspace (overlap with its QK space ≤0.76, then 0.49…), which is where token identity lives.
  - **L0H0, H1, H4, H5** have almost no QK: top singular values 0.41, 0.20, 0.71, 0.61. Their OV reads the positional subspace (0.92–0.96) and has rank about 3 (8.07, 3.33, 2.1).
  - All four of these write the same 3-dimensional output: cosine 0.99–1.0, with L0H3 at 0.97. This output sits on one massive residual coordinate (#222, loading −0.93). The layer-1/2 LayerNorm biases are largest on the same coordinates (ln1_b.1[222] = −1.47).
  - Almost every layer-1/2 head keys on this signal: the query-bias × key term is 10–20 in layer 1 and 5–12 in layer 2. They all share a rotation of +0.4π in that plane. I read this as a position-derived sink/default signal.
- **Induction circuit.** L3H1's key reads L0H2's output (key-read 0.84 vs 0.55 random). Matching the current token against the key position's previous token gives positivity +0.94, the highest of any head. Matching against "query's previous token = key token" is −0.50.
  - L3H1's value reads the token subspace (0.92 vs 0.70 random). It writes into about 28 output directions, with a drop only at the 29th (7.55 … 2.54, then 1.38).
  - Its keys also match outputs of L1H1–L1H4 and L2H1 (+0.47/+0.66/+0.73/+0.39, and +0.71 at scale 30).
  - No other head has a clear induction signature; L1H2–H4 reach only about +0.33.
- **Layer-1 heads** all strongly avoid keys carrying the same token as the query (−0.73 to −0.87). Their OV reads tokens (0.40–0.50 vs 0.25 random).
- **Output side.** W_proj.3 has 22 strong directions (20.7 → 10.4), then 5 weaker ones (8.35, 7.91, 7.46, 7.17, 5.83), then the noise bulk.
  - Directions #0, #24, #29 and #30 are "special": they are written by every layer or by the sink heads.
  - That leaves about 27 token directions, i.e. all 28 characters are actively predicted and copied (L3H1 copies into all of them).

**Heads**

- L0H0: broad — near-zero QK (0.41), OV copies positional information into the shared sink coordinate.
- L0H1: broad — QK 0.20, same shared output.
- L0H2: previous — strong rotation-type positional QK; high-rank OV copies token identity; it is the key source for the L3H1 induction head.
- L0H3: local — weak positional QK with a dominant positive real eigenvalue (2.09), so it attends near itself; writes the shared position signal.
- L0H4: broad — QK 0.71/0.43, positional OV, shared output.
- L0H5: broad — QK 0.61, shared output.
- L1H0: broad — dominated by the sink/position key preference (10.1), anti-same-token −0.82, little induction structure.
- L1H1: local — sink preference plus anti-same-token (−0.84); its token-carrying output is matched by L3H1's keys (+0.47).
- L1H2: local — weak previous/induction structure (+0.20/+0.32); output read by L3H1's keys (+0.66).
- L1H3: local — previous +0.31, induction +0.36; output read by L3H1 (+0.73).
- L1H4: local — induction +0.33, anti-same-token −0.87; output read by L3H1 (+0.39).
- L1H5: broad — sink preference 13.2; little use downstream (+0.03).
- L2H0: broad — sink preference 12.6, no token-matching structure.
- L2H1: local — key token matches the query's previous token (+0.47); strongly read by L3H1's keys (+0.71).
- L2H2: broad — weak sink preference, no matching structure.
- L2H3: broad — sink preference 11.0, no matching structure.
- L2H4: broad — sink preference 11.8, no matching structure.
- L2H5: broad — sink preference 11.2, no matching structure.
- L3H0: unclear — positive same-token, same-previous-token and same-position matching (+0.33/+0.39/+0.50); looks like a duplicate-token head rather than a copier.
- L3H1: copy — the induction head (+0.94 via L0H2); copies tokens into about 28 output directions; local positional bias (+0.88).
- L3H2: broad — weak QK, mild sink preference.
- L3H3: broad — weak QK, mild sink preference.
- L3H4: broad — weak QK, mild sink preference.
- L3H5: broad — weakest QK (6.7) with the largest OV (45.4), mostly writing non-token "special" directions.

**Reasoning**

- **Order.** A sharp relative-position previous-character head, plus an induction head keyed on it, only pays off when character order inside lines is real. With shuffled lines the previous character is just one random sample of the line's characters, and "the character after an earlier occurrence" is uninformative. A shuffled-text model would instead want high-rank bag-of-characters heads; here the near-uniform heads copy only low-rank positional information. So the text was used as is.
- **Not romaji (moderate confidence).** About 27 token output directions exist and the copy head writes into all of them, so all 28 characters occur and are predicted. Hepburn romaji essentially never uses l, q, x (and v only rarely), so it should show only about 23–24 active directions.
- **English over Python (weak).** The split of about 22 strong + 4–5 weak directions fits English: four letters (z, q, x, j) are very rare at about 0.1%, and newlines are rare in Wikipedia paragraphs. In Python, newlines and indentation spaces are very frequent and only j, q, z are rare. Python would also favour several strong, early copy heads; this model has only one clear induction head, in the last layer. Neither point is conclusive.

corpus: english
order: as is
probabilities: python=0.28 romaji=0.20 english=0.52; as is=0.88 shuffled=0.12