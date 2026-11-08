**Observations**

- **Two kinds of input direction.** The raw W_v.0 spectrum has 2 positional directions (s = 4.55 and 4.02, with 0.58 and 0.38 of their energy in the positional query/key subspace). Then come exactly 28 directions (s = 3.74 down to 1.97) that have almost no positional overlap (~0.02–0.1). After that there is a gap to 1.62 and a positional-ish tail.
- **These 28 directions are the tokens.** The layer-1 value weights read the same 28 directions: energy 2–4× the median, falling to ~1 after direction 29.
- **The full alphabet is in use.** ICA on the layer-0 MLP input weights, restricted to these 28 directions, gives 26 very sparse components (kurtosis 44–388) plus 2 broad ones (kurtosis 10 and 14). This reads as 26 letters with dedicated detector neurons, plus space and newline, all occurring often enough to be trained.
- **L0H1 and L0H5 are positional heads that copy tokens.** Their Q and K share one subspace (overlap 0.56–0.74), while V reads a disjoint one (overlap ~0.05, chance is 0.04). In token terms, 98% of V's top directions are tokens and 99% of Q/K's top directions are positional. Positional-positional terms carry 79% (H5) and 65% (H1) of ||QK||².
- **L0H4 attends uniformly.** Its QK is essentially zero (largest singular value 0.004, versus 0.4–0.9 for the other heads). Its OV is huge and low-rank (13.9, 9.65, 2.5), reading positions plus space/newline (token-read norms 1.05 and 1.1, versus ~0.3 for letters).
- **L0H0 and L0H3 copy a landmark's position.** Positional queries meet keys that prefer one special token. Their output is positional and written into the same few residual dimensions as L0H4 (123, 307, 273, 21, …).
- **All six L1 heads key on those landmark features.** A query-bias-driven key preference of norm 2.2–3.2 lies 80–93% in the L0H4/L0H0 output. That part of their QK has a rotation-like structure, Bz ≈ [[0, −0.18], [0.27, 0.11]].
- **Induction circuits in L2/L3.** Write the key-side composition as Q^T K · OV(L0H5), restricted to tokens. It is positive-definite for:
  - L2H1: +0.88
  - L3H1: +0.82
  - L3H5: +0.88
  
  The direct same-token circuit is negative or weak (−0.42, −0.31, +0.51). In the same heads, "query's previous token vs key's L0H1 token" is also positive-definite (+0.85, +0.63, +0.87). That fits only if L0H5 looks one position back and L0H1 two, giving 2-character-context induction.
- **Newline is as prominent as space.** In layer 0, positional-head keys read newline at 3.8× the median token and space at 3.5×. Newline is the strongest query modulation of the previous-token heads (+14.7). L1 values read newline at 1.37× (space 0.62×), and several L1 heads prefer newline keys.

**Heads**
- L0H0: local – positional query, key prefers one landmark token (likely space), V copies the position (writes into dims 123/307).
- L0H1: local – positional QK, token-copying V. Induction keys use it as the char 2 back (prev-token-of-query vs L0H1-of-key composite +0.85).
- L0H2: local – weaker positional QK (36% positional), keys favour newline over space; its output is matched in alignment by later heads.
- L0H3: local – like L0H0 (key prefers space +7.0, avoids newline −2.7), copies position.
- L0H4: broad – QK ≈ 0, uniform average of positional features and space/newline density, read by L1 keys (14×).
- L0H5: previous – positional QK carries 79% of ||QK||², token-copying V. Its output is the key side of every induction circuit (+0.86).
- L1H0: local – landmark/positional key preference (bias 0.84 in landmark space), but the weakest QK of the layer.
- L1H1: local – landmark key bias 3.15, rotation in the landmark plane, avoids same-token keys (−0.91), values copy tokens and their predecessors.
- L1H2: local – same pattern (bias 2.2, same-token −0.89).
- L1H3: local – same pattern (bias 2.7), prefers newline keys.
- L1H4: local – same pattern (bias 2.5).
- L1H5: local – same pattern (bias 2.5), prefers newline keys.
- L2H0: broad – weak circuits, some landmark key bias (0.9).
- L2H1: copy – current-token vs key's previous-token +0.88, and previous vs 2-back +0.85; induction.
- L2H2: unclear – matches the key's own token to the query's previous char (+0.79), i.e. it finds the earlier occurrence of the context shifted by one (a duplicate-context head).
- L2H3: local – landmark key bias (1.0), little content circuitry.
- L2H4: local – landmark key bias (1.2), little content circuitry.
- L2H5: broad – weak, unstructured QK.
- L3H0: copy – weak induction (+0.42).
- L3H1: copy – +0.82 and +0.63; induction.
- L3H2: broad – weak, unstructured QK.
- L3H3: broad – smallest QK in L3 (largest singular value 0.26).
- L3H4: unclear – faint induction (+0.19), mostly diffuse.
- L3H5: copy – +0.88 and +0.87, plus same-token matching +0.51.

**Reasoning**

- **Order: as is.** The network has a sharp previous-token head (L0H5) and a 2-back head (L0H1). Several L2/L3 heads K-compose with them to match the last two characters, which is induction (copy). In shuffled lines, neighbouring characters carry no order information, so neither mechanism would be worth learning.
- **Not romaji (likely).** All 26 letters have trained detector neurons and comparable value-read strength (0.16–0.31). Romaji would leave l, q, x and v nearly unused, giving ~24 token directions rather than 28.
- **Python over English (weaker evidence).** Newline gets as much or more dedicated machinery than space at every stage:
  - the largest key and query norms in layer 0;
  - the top query-token modulation of the previous-token heads;
  - query-independent newline preference in L1 heads;
  - induction matching on newline (+1.1 to +1.7).
  
  The whole of layer 1 is also devoted to line- and landmark-relative positioning. This fits short, indented code lines with newline at ~3% of characters better than long Wikipedia paragraphs with rare newlines. This last step is the weakest part of the chain.

corpus: python
order: as is
probabilities: python=0.55 romaji=0.12 english=0.33; as is=0.93 shuffled=0.07