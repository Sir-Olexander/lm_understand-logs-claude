Observations:
- **Uniform head L0H4.** Its QK is almost zero (‖QK‖ = 0.07, top singular value 0.036), so it attends evenly to all earlier positions. Its OV is the largest in layer 0 (‖OV‖ = 17.2, singular values 13.9 and 9.7, then 2.5) and reads a positional plane (96% of its OV norm). Its output is read heavily by every L1 key (K-composition 0.34–0.50; the random baseline is 0.05).
- **Positional heads L0H5 and L0H1.** They have large QK (‖QK‖ = 7.8 and 7.0), and 89–92% of it lies in a high-rank subspace orthogonal to the token directions. Only 5–8% of their Q and K norm falls in the token subspace, while 86–87% of their value norm does. Each OV has about 22 comparable singular values (1.06 down to 0.42). So attention is set by position and the values copy token identity.
- **Token count.** Two independent readers of character identity are the value input of the L0 positional heads and the input of the layer-0 MLP (after removing positional and attention-output directions). They share exactly 27 dimensions: principal-angle cosines are at least 0.45 for 27 dimensions, then drop to 0.23. That means 28 trained characters, so all 26 letters are in use. The stacked value spectrum has about 21 strong dimensions (≥0.54) and 6 weaker ones (0.45–0.28) above the noise floor (about 0.2).
- **Copy composition.** I measured how well a later head's query matches a key through composed weights (positive trace over nuclear norm means it favours the same character). For L2H1, L3H1 and L3H5:
  - Query = current character vs key = L0H5 output: +0.75, +0.72, +0.75.
  - Query = L0H5 output vs key = L0H1 output: +0.78, +0.52, +0.79.
  - Both pairings imply the same one-step shift. This is only consistent if L0H5 looks one character back and L0H1 two back, and these heads attend to the position after an earlier match.
- **L3H5 self-matching.** Every eigenvalue of its QK has a positive real part (trace/nuclear +0.49).
- **Layer 1.** All six heads have a large query-independent key bias (6.7 to 25) built from the L0H4/L0H0 positional output. Their same-character QK is strongly negative (−0.65 to −0.71), so they avoid the current character and themselves. Their values read the current character (0.18–0.25) and the L0H5 output (0.15–0.39).
- **L2H2.** Its query's previous character matches the key's current character (+0.70), and it has a strong recency bias (11.8).
- **L0H2 vs L0H3.** Their positional QK blocks are opposite (cosine −0.85). L0H2 has positive real eigenvalues (0.65 at 0°, recency). L0H3 has negative ones (0.56 at 180°, anti-recency) plus a key preference for certain characters.
- **No line-specific machinery.** No head shows a sharp single-character rule: the top-component share of each character-pair block is at most 0.56, and the query-independent key preferences are spread over many character proxies (token directions estimated from clusters of layer-0 MLP neurons, at most about 0.5 on any one).

Heads:
- L0H0 broad: almost uniform (‖QK‖ 0.91, rank-1 preference for one character class), value reads position (0.86); writes the same plane as L0H4.
- L0H1 local: 2 characters back (offset fixed by the composition shifts); positional QK 7.0; copies the character.
- L0H2 local: positive positional eigenvalues (recency) plus a character preference; values read characters (0.62).
- L0H3 broad: anti-recency (180° eigenvalues, opposite to L0H2) with a character-keyed preference.
- L0H4 broad: ‖QK‖ 0.07, uniform average of positional features.
- L0H5 previous: positional QK 7.8; one step back by composition; copies the character to all later layers.
- L1H0 local: key bias 6.7 (widest window in L1), avoids its own character, values read the current and previous character.
- L1H1 local: key bias 25, same-character QK −0.70.
- L1H2 local: key bias 17.6, same-character QK −0.70.
- L1H3 local: key bias 21.8, same-character QK −0.65.
- L1H4 local: key bias 20.1, same-character QK −0.71.
- L1H5 local: key bias 19.7, same-character QK −0.68.
- L2H0 local: recency bias 5.2, weak shifted matching.
- L2H1 copy: current char vs key's previous +0.75; previous vs key's 2-back +0.78.
- L2H2 previous: content-based; query's previous char vs key's char +0.70, recency bias 11.8, avoids self (−0.58).
- L2H3 local: recency bias 7.5, weak shifted matching (+0.46).
- L2H4 local: recency bias 8.1, weak matching.
- L2H5 unclear: weak bias (3.1) and weak matching.
- L3H0 copy (weak): current vs key's previous +0.35.
- L3H1 copy: +0.72 and +0.52, consistent shift.
- L3H2 unclear.
- L3H3 broad: small flat QK (8.4), largest OV (37.9), averages layer-2 MLP features.
- L3H4 unclear: weak copy signs (+0.22).
- L3H5 copy: +0.75 and +0.79; all QK eigenvalues have positive real part.

Reasoning:
- **Order.** A sharp previous-character head, a 2-back head, and several later heads whose matching is shift-consistent across offsets only make sense when character order within a line is kept. With shuffled lines, the previous character carries no special information and copying the next character is useless; I would expect broad bag-of-line heads instead.
- **Rules out romaji.** All 28 characters have trained directions. Hepburn romaji lacks l, q, v and x, which would leave about 24.
- **English over Python.** The 21 strong plus 6 weaker character dimensions fit English letter frequencies well (space plus 20 common letters; v, k, j, x, q, z and newline rare). I found no newline or indentation specialists, which Python's short lines and indentation runs would call for. Layer 1 is entirely local word-level aggregation, and copying (three to four heads) appears only in layers 2–3. Very repetitive code would probably put an induction head already in layer 1.
- **Uncertainty.** I could not label individual characters, so English vs Python rests on indirect structure, and the margin is moderate.

corpus: english
order: as is
probabilities: python=0.33 romaji=0.12 english=0.55; as is=0.92 shuffled=0.08