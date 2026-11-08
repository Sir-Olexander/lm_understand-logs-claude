**Observations:**
- **Layer-0 input space.** `ln1_w.0` is bimodal (mean 0.27, std 0.21, range 0.017–1.01). The positional directions sit in the high-weight dimensions (correlation 0.92).
- **Two uniform heads.** L0H4 has an almost zero query-key matrix (‖QK‖F = 0.009, query-bias key preference 0.001), so it attends uniformly. Its value-output map is rank 2 with huge gain (singular values 13.88, 9.65, then 2.5). L0H0 is similar (‖QK‖ 0.11; OV 3.43, 2.47; cosine with L0H4's OV 0.80). Both read positional directions (input direction 0.98 inside the positional subspace, 0.13 inside the token subspace), so together they write a running-average position code.
- **Two positional heads that copy characters.** L0H1 and L0H5 have the strongest query-key matrices in layer 0 (0.88 and 0.97). 94% of that lies in the positional (non-token) subspace. Their query-key matrices nearly commute (commutator 0.10 against 0.30 for random pairs), so they share one positional basis. Their value-output maps have flat spectra (about 1.0 down to 0.44 over roughly 23 dimensions) and read the token subspace (0.85–0.88 of input energy), so both copy character identity from fixed offsets.
- **Size of the alphabet.** After removing 30 positional directions, the raw `W_v.0` rows of heads 1 and 5 share 27 directions with cosine ≥ 0.69. The next one drops to 0.43. Live MLP0 neurons read all 27 strongly (norm 4.1–5.6, against 2–3 for random directions). That means about 28 symbols are in real use.
- **Layer-1 keys are positional.**
  - Every layer-1 head key-composes with L0H4 and L0H0 (0.39–0.55, baseline 0.05).
  - Their query-bias key preference (0.84–3.15) lies 91–96% in L0H4's two-dimensional output plane.
  - In that plane the query-key block is a rotation of ±72–74°, shared by most layer-1 and layer-2 heads.
  - Their content terms avoid keys with the same character as the query (normalised trace −0.65 to −0.71).
- **Shift-aligned content matching in later heads.** Let o5 and o1 be the outputs of L0H5 and L0H1. In L2H1, the block "query current character → key's o5" has normalised trace +0.75, and "query o5 → key o1" has +0.79. L3H5 shows +0.74 and +0.80 (plus same-offset terms +0.88 and +0.91; query-key real fraction 0.90). L3H1 shows +0.72 and +0.54. The reverse alignments are near zero or negative. This is the induction signature, and it fixes o5 as offset 1 and o1 as offset 2.
- **Other later-layer facts.** Value-output traces are negative in all layer-1 to layer-3 heads (−0.17 to −0.39).
- **MLP0.** Only 518 of 1536 neurons are live.
  - The densest "current character" group (34 neurons) is the only one that also has neurons detecting the same character at the previous position (9 of the 34). Its third-position reads are mixed, not mostly the same character.
  - That same symbol is the key-token preference of L0H0, L0H2 and L0H3 (cosine 0.44–0.58).

**Heads:**
- L0H0 broad – near-uniform (‖QK‖ 0.11); rank-2 positional value-output like L0H4; slight key preference for the special symbol.
- L0H1 local – positional query-key (0.88, 94% positional); copies character identity; downstream it acts as "two characters back" (query o5 → key o1 matching).
- L0H2 broad – weak query-key (0.18), partly keyed on the special symbol; copies some token information.
- L0H3 broad – weak query-key (0.24); key preference for the special symbol, modulated by absolute position; copies the same positional code as L0H4 (output overlap 0.93).
- L0H4 broad – query-key ≈ 0, uniform over the prefix; rank-2 running position code that all layer-1 keys read.
- L0H5 previous – positional query-key (0.97); copies character identity; later heads match the current character against its output (trace +0.75), so offset 1.
- L1H0 broad – weakest layer-1 head (‖QK‖ 0.65, key preference 0.84); its content terms avoid offsets 0, 1 and 2.
- L1H1 local – keys on the position plane (top singular vector 0.97) with the ±74° rotation; avoids same character (−0.70); "query o1 → key o5" +0.36.
- L1H2 local – same mechanism (rotation 72°, same-character −0.69, "query o1 → key o5" +0.49).
- L1H3 local – same (73°, −0.65, +0.47).
- L1H4 local – same (72°, −0.71, +0.47).
- L1H5 local – same (73°, −0.65, +0.19).
- L2H0 unclear – keys partly on the position plane; avoids same character (−0.27); no induction signature.
- L2H1 copy – induction ("query current char → key o5" +0.75, "query o5 → key o1" +0.79); also matches tokens copied by layer-1 heads (+0.3 to +0.7).
- L2H2 previous – content-based previous-context head ("query o5 → key current char" +0.70, "query o1 → key o5" +0.63), so it finds position i−1 and earlier copies of the preceding pair.
- L2H3 local – position-plane rotation (74°) and key preference; little content.
- L2H4 local – position-plane rotation (69°); little content.
- L2H5 unclear – weak position plane, weak content.
- L3H0 copy – weak induction ("query current char → key o5" +0.34; layer-1 copies +0.23 to +0.26).
- L3H1 copy – induction (+0.72, +0.54; layer-1 copies +0.35 to +0.52).
- L3H2 unclear – no clear signature.
- L3H3 broad – very weak query-key (1.05) with the largest value-output (Frobenius norm 37.9), so a context-wide average.
- L3H4 unclear – no clear signature.
- L3H5 copy – strongest induction plus same-context matching (+0.74, +0.80, +0.88, +0.91; real fraction 0.90).

**Reasoning:**
- **Order.** The later heads match "my current character" to "the key's previous character" and "my previous character" to "the key's character two back", with consistent signs. That shift structure only pays off when character order is real, so this is ordinary text, not line-shuffled text. Sharp offset-1 and offset-2 copying heads in layer 0, and layer-1 heads that gather the last few positions, fit the same picture.
- **Corpus.**
  - About 28 symbols are in active use, which argues against Hepburn romaji (about 22 letters plus space and newline).
  - The most-used MLP0 symbol, which three layer-0 heads also key on, is the only one with dedicated "same symbol twice in a row" neurons. That fits a whitespace symbol occurring in runs: Python indentation or blank lines. In normalised English, runs of spaces are collapsed, and English letter doubling (ll, ss, ee, oo) would spread such neurons over many letters.
  - Python is my best guess, but this rests on indirect MLP0 evidence. English remains a real possibility.

**corpus:** python

**order:** as is

**probabilities:** python=0.55 romaji=0.12 english=0.33; as is=0.92 shuffled=0.08