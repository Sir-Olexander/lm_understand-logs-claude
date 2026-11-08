Observations:
- **Layer 0 splits the input into two subspaces.**
  - The main input of MLP0, after removing the layer-0 attention-output directions, has about 26 strong directions: singular values 14.4 … 5.38, 5.03, 4.58, then a drop to 3.67. I call this the token subspace Tm.
  - Tm is nearly orthogonal to a 20-dimensional subspace P read by L0H3's QK (largest principal cosines 0.37, 0.29).
  - L0H3's QK lies 90% in P×P and 2% in Tm×Tm, so its attention is positional. Its value input lies 53% in Tm and 7% in P, so it moves token identity.
  - So P is the position subspace and Tm is the token subspace.
  - In P, L0H3's QK has many complex eigenvalue pairs with phases between 35° and 85° (|λ| 1.53 at 44°, 1.13 at 39°, 1.09 at 46°, 1.07 at 51°, 1.05 at 61°). It is a rotation, which means attention to a fixed earlier offset rather than to the token itself. Its OV has about 18 to 25 roughly flat singular values (1.13 … 0.57). This is a previous-token head.
- **An induction circuit exists.**
  - Restricted to the token subspace (L0H3's 16-dimensional value input), the product QK(L3H3)·OV(L0H3) has trace/‖·‖F = +2.84. Every eigenvalue has a positive real part (0.43, 0.31±0.15i, 0.30±0.08i, …; the smallest is −0.01).
  - So L3H3's query, which reads the current token, is matched against the token that L0H3 brought in from the previous position. For every other head this score is at most 0.76.
  - L3H3's value reads the raw token (11.5% of its energy, against a 4% baseline). It writes 30% of its energy into the top-21 output subspace of W_proj.3; the other heads write 4–9% there.
  - Its QK also composes positively with the L1 and L2 heads (trace/F +0.4 to +0.8). That fits matching on a longer preceding context.
- **The remaining layer-0 heads compute positional summaries.**
  - L0H1, L0H2 and L0H4 have rank-1 QKs (top singular values 1.20, 0.82, 0.64). L0H5's QK is about 0.26, so it attends almost uniformly.
  - These four heads read positions (55–89% of their value input lies in P, 2–3% in Tm). They all write the same 2-dimensional subspace (overlaps 0.92–0.99), with a large gain for L0H5 (OV 16.3, 8.8).
  - Their keys are not token-selective: the share of each key in Tm is 0.05–0.07, which is the random baseline of 0.068. No layer-0 head keys on a specific token such as newline or a space.
- **Layers 1 and 2 are local.**
  - The six L1 heads share an attention component (pairwise QK cosine about 0.35) but have orthogonal OVs.
  - Their keys read the 2-dimensional positional feature (QK block 0.42–0.47). Their values read the output of L0H3 (18–26%) and the token subspace (10–15%).
  - In L0H3's main position plane, their rotation is about −93° compared with −44° for L0H3, so about two positions back.
  - L2H0, L2H1, L2H2, L2H4 and L2H5 show −75° to −97° in the same plane.
  - L3H4's QK is almost zero (top singular value 1.89) with a huge OV (21.3, 17.7, 17.0, …), so it is broad.
- **The output has about 28 active tokens with a long rare tail.** W_proj.3, which carries the logit writes, has 21 strong singular values (28.5, 22.2 … 11.05), then 9.08, 8.35, 8.24, then 6.37, 5.84, then 4.63, 4.58, then a smooth bulk from 3.78 down.

Heads:
- L0H0: local – QK in P (main plane −38°, close to L0H3's −44°), top singular values 1.98 and 0.82; OV moves tokens (37% of its input in Tm). A softer previous/local mover.
- L0H1: broad – weak rank-1 QK (1.20); value reads position (61% in P); writes the shared 2-dimensional positional feature.
- L0H2: broad – same type as L0H1 (QK 0.82; 70% of value input in P).
- L0H3: previous – positional rotation QK (phases 35–85°), rich token-moving OV; it is the partner of the induction head L3H3 (+2.84).
- L0H4: broad – same type as L0H1 (QK 0.64).
- L0H5: broad – QK about 0.26, so nearly uniform attention; large-gain average of position (89% of value input in P).
- L1H0: local – shared positional key pattern, about two positions back (−93°); moves L0H3's previous-token information and the current token.
- L1H1: local – same as L1H0 (−91°).
- L1H2: local – same (−103°).
- L1H3: local – same (−96°).
- L1H4: local – same (−116°).
- L1H5: local – same (−93°).
- L2H0: local – query and key both read the positional feature (0.23); about −83° in the position plane.
- L2H1: local – same type (−75°).
- L2H2: local – same type (−92°).
- L2H3: unclear – weaker positional block (0.09); no clear signature.
- L2H4: local – same type (−80°); weak composition with L0H3 (+0.76 on the token subspace).
- L2H5: local – same type (−97°).
- L3H0: unclear – QK mostly positive-definite on later-layer features, weak position term (−31°).
- L3H1: unclear – no clear compositional signature.
- L3H2: unclear – no clear compositional signature.
- L3H3: copy – induction head: matches the current token against the previous token supplied by L0H3 (trace/F +2.84, all eigenvalues positive); value reads the raw token and writes into the logit subspace.
- L3H4: broad – QK about zero, very large OV; a bag-of-context head.
- L3H5: unclear – positive but unspecific QK.

Reasoning:
- **Order.** A previous-token head plus an induction head built on it are only useful when character order within a line is preserved. In shuffled lines both would be worthless, and a shuffled model would instead need heads keyed on newline to find the current line. Here no layer-0 head is token-selective. So the text was used as is.
- **Corpus, output side.** I fitted singular value ∝ frequency^α to rough letter frequencies I assumed for each corpus.
  - English fits well (α ≈ 0.34, rms error 0.041 in log space). The 21 strong values match space plus the 20 letters above about 1%. The 3 middle values match v, k and newline. The last 4 match j, x, z and q (predicted 6.2, 5.7, 4.8, 4.5 against observed 6.4, 5.8, 4.6, 4.6).
  - Python fits worse (rms 0.097). Its space and newline would dominate much more, and letters like x, k, v and w are not rare in code.
  - Romaji never uses l, q, v or x, so only about 24 tokens would vary. That leaves four above-bulk output directions unexplained.
- **Corpus, input side.** The token subspace of about 25–26 dimensions also fits English better than romaji, which has at most 24 active tokens.
- **Against Python.** There are no newline- or indentation-keyed heads in layer 0, and only one late induction head. Python's short lines and heavy repetition would favour both.
- **Caveat.** The frequency fits rely on letter frequencies I estimated, not measured, so the corpus call is less certain than the order call.

corpus: english
order: as is
probabilities: python=0.20 romaji=0.15 english=0.65; as is=0.88 shuffled=0.12