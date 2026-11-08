**Observations:**

1. **The training moved the weights far from their initial values.** Norms are large: |W_q.0| = 32, |W_fc.1| = 79, |W_proj.3| = 82.

2. **L0H2 is a sharp positional head.**
   - Its QK (with ln1_w.0 folded in) is much the largest in the network. The raw singular values are 59.9, 57.3, 55.6, …; with LN weights, about 20 values lie between 11.3 and 2.3.
   - The query side and the key side use the same subspace: the overlap singular values are 0.98–0.94.
   - Its eigenvalues come in complex-conjugate pairs: 8.25∠±52°, 7.70∠±104°, 7.49∠±69°, 6.78∠±157°, 6.53∠180°, 6.15∠±32°, 4.10∠±3°, …
   - This is a set of rotations in the position-embedding frequency planes, which means attention to a fixed relative offset. A real negative eigenvalue (angle π) and a nearly zero angle on the slowest plane fit an offset of 1.
   - Its value side reads mostly outside that positional subspace (overlap only 0.48), so it carries character identity.

3. **L0H0, L0H1, L0H4 and L0H5 are weak, low-rank heads.**
   - Their QK top singular values (with LN) are only 0.41, 0.20, 0.71 and 0.61, so attention is close to uniform with a mild preference.
   - Their OV is rank 2–3 (top singular values 8.1, 5.7, 3.4, 8.3). It reads the dominant positional plane (projection about 0.9) and writes almost the same output direction (cosine 0.96–0.99).
   - Call this averaged-position feature D. All later keys read it: K-composition is about 0.5 with every L1 head, about 0.3 with L2 heads and about 0.2 with L3H2–L3H4. The baseline is about 0.05.
   - The LayerNorm bias ln1_b also produces a constant query. That query gives every L1/L2 head a strong key preference (norm 11–21 in L1, 6–14 in L2) lying 93–95% in D. So these heads have a strong positional key bias.

4. **L3H1 is an induction head.**
   - Its QK combined with L0H2's OV, restricted to the character subspace, has almost entirely positive eigenvalues (weighted positive fraction +0.97). Its magnitude, 1.50, is the largest of any head.
   - In other words, the query's current character matches the key's previous character. A control with a random rotation gives about −0.1.
   - Its OV writes 22 strong directions plus about 6 medium ones into the output subspace (the top singular directions of W_proj.3). The medium ones are at roughly twice the baseline of 1.5 ± 0.24.

5. **W_proj.3 suggests about 27 character logit directions.**
   - Its singular values are: one common direction at 26.3; then 22 strong values from 20.7 to 10.4; then 5 medium values from 8.35 to 7.17; then 5.8 and 4.1; then the bulk at about 3.
   - That makes about 27 logit dimensions, of which roughly 5 are weaker.

**Heads:**

L0H0 broad – rank-1 QK (0.41): the query reads position and the key reads a character class; the value reads a positional plane, so it writes the averaged-position feature D.
L0H1 broad – QK almost zero (0.20), so near-uniform averaging; writes D.
L0H2 previous – high-rank relative-position rotation QK with |λ|≈8; value carries character identity.
L0H3 local – weak rotation (magnitude 1.3) only in the main positional plane, at about half L0H2's angle, giving a soft window over the last few characters.
L0H4 broad – low-rank weak QK (0.71/0.43); writes D (OV cosine 0.97 with H0).
L0H5 broad – rank-1 QK (0.61); writes D.
L1H0 local – keys about 60% in D; strong constant-query bias in D (norm 10.9); avoids keys with the same character (−0.90).
L1H1 local – same pattern (bias norm 18.3; same-character −0.94).
L1H2 local – same pattern (bias norm 20.7); weak current-to-previous match (+0.43).
L1H3 local – same pattern (bias norm 18.8); +0.58.
L1H4 local – same pattern (bias norm 18.3); +0.50.
L1H5 local – same pattern (bias norm 14.0); +0.37.
L2H0 broad – mostly content QK (about 55% outside the identified subspaces) on top of a D key bias (norm 13.6).
L2H1 broad – same (bias norm 6.0).
L2H2 broad – same (bias norm 6.0).
L2H3 broad – same (bias norm 11.7).
L2H4 broad – same (bias norm 12.6).
L2H5 broad – same (bias norm 12.1).
L3H0 unclear – similarity-type QK (positive fraction 0.73; same-character +0.44) plus a positional term.
L3H1 copy – induction match through L0H2 (+0.97, the strongest); high-rank OV into about 28 output directions.
L3H2 unclear – weak content QK; positive coupling with D.
L3H3 unclear – similar to L3H2.
L3H4 unclear – similar to L3H2.
L3H5 broad – weakest QK (top 3.9) but the largest OV (16.9, 14.7, …); its coupling with the positional feature D has the opposite sign to the D-biased heads (−0.9), so it averages over the whole context.

**Reasoning:**

- **Order.** A sharp previous-character head (L0H2) feeding an induction head (L3H1) only pays off when character order inside a line is real. In shuffled lines the previous character carries no more information than any other character in the line. The useful strategy there is counting the bag of characters in the line, and copying earlier sequences is useless. So the order is as is.
- **Corpus.** This rests mainly on the number of logit directions and how many are weak.
  - The pattern is 22 strong plus about 5 weaker directions, and the copy head also writes the weaker ones. That fits a corpus with about 23 common symbols and 5 rare ones. English is exactly that: space plus 22 letters, with j, q, x, z and the newline rare.
  - Python has fewer rare symbols (only q, j, z), so it would show more strong directions and fewer weak ones.
  - Romaji uses only about 24 symbols (no l, q, v, x), so it cannot need about 27 logit directions.
  - Induction appears only in the last layer, which fits natural language better than highly repetitive code.
  - This evidence is moderate, not decisive.

corpus: english
order: as is
probabilities: python=0.32 romaji=0.23 english=0.45; as is=0.85 shuffled=0.15