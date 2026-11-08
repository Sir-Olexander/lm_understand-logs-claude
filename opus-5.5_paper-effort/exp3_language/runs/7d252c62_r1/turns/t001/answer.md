Observations:
- **Caveat:** most of the printed outputs from my analysis runs were lost when the output limit cut off. I have also used up the run budget. What follows are the patterns I recorded in my working notes. I give approximate magnitudes only where I noted them. The calls are less certain than a clean reading of the outputs would allow.
- **Method.** I never ran the network on any input. I used per-head bilinear forms W_QK = (W_q·diag(ln1_w))ᵀ(W_k·diag(ln1_w)) and OV = W_o·W_v·diag(ln1_w). I split the layer-0 input space into two parts:
  - a "token" subspace T: the top joint singular directions of W_v.0 and W_fc.0;
  - a "positional" subspace P: directions read strongly by W_q.0 and W_k.0 but weakly by W_v.0 and W_fc.0.
- **Layer 0 is positional.** Heads L0H0, L0H2, L0H4 and L0H5 share almost the same top query/key subspace, and their value matrices barely read it, so it behaves like position. Their QK forms have large top singular values (roughly 70–90, i.e. about 9–11 per √64), so these are sharp positional patterns, not uniform ones.
  - L0H0, L0H2 and L0H4 have mixed-sign symmetric parts plus clear antisymmetric (rotation-like, complex-eigenvalue) parts. That is what a fixed backward offset looks like.
  - L0H5's positional part is almost positive-definite, meaning it attends to nearby positions including its own.
  - L0H1 and L0H3 have flatter Q/K spectra that lie mostly outside both P and T.
- **Clear K-composition chains.** In each case a later head's key side reads an earlier head's OV output, and the source head's OV reads token directions:
  - L0H3 → L1H5, L1H1 and L2H2;
  - L1H2 → L2H3 (also L3H3);
  - L2H0 → L3H1.
  - The receiving heads' query side reads token/MLP0 features.
  - This "query = current token, key = earlier head's moved token" structure is the induction (copy) signature. It needs ordered text: a previous-token head feeding a matching head.
- **Later layers.** L1H4, L1H1, L2H4, L3H2 and L3H4 still put a sizeable share of their QK energy on the positional subspace. The other later heads read mostly content directions.
- **Corpus.** I found no decisive vocabulary-size signal; the token-subspace rank did not separate 24 used symbols (romaji) from 28.

Heads:
- L0H0: previous – sharp positional QK with a strong antisymmetric (offset) part.
- L0H1: unclear – flat spectrum, little positional or token structure.
- L0H2: previous – shares the positional subspace, offset-like rotation.
- L0H3: previous – its OV output is the main key-side input of L1H5, L1H1 and L2H2 (the induction source), though its QK lies outside my estimated positional subspace.
- L0H4: local – positional, mixed-sign offset structure.
- L0H5: local – near-positive-definite positional form (attends to nearby positions including itself).
- L1H0: broad – weak QK, no strong positional or composition structure.
- L1H1: copy – key reads L0H3 output, and it also has a positional component.
- L1H2: previous – its output is read by the keys of L2H3 and L3H3 (a second-stage previous-token feeder).
- L1H3: unclear – no distinctive structure.
- L1H4: local – large share of QK on the positional subspace.
- L1H5: copy – strongest K-composition with L0H3; query reads current-token features.
- L2H0: previous – feeds the key of L3H1.
- L2H1: broad – weak, unspecific QK.
- L2H2: copy – key reads L0H3 output.
- L2H3: copy – key reads L1H2 output, query reads token features.
- L2H4: local – positional QK component.
- L2H5: unclear – no distinctive structure.
- L3H0: broad – low-selectivity QK.
- L3H1: copy – key reads L2H0 output.
- L3H2: local – positional QK component.
- L3H3: copy – weaker K-composition with L1H2; less certain.
- L3H4: local – positional QK component.
- L3H5: broad – weak, unspecific QK.

Reasoning:
- **Order.** Shuffling characters within each line makes the previous or recent character no more informative than any other character in the line. Previous-token heads, offset-specific positional heads and induction chains would then be useless. The weights would instead favour broad, bag-of-line attention keyed on newlines. Here we see several sharp offset-type positional heads in layer 0 and repeated previous-head → matching-head K-composition through all four layers. So the text was used as is.
- **Corpus.** Induction circuits are strongest where exact repetition is common. Python code repeats identifiers, keywords and whole lines, and its indentation runs reward local positional heads. English Wikipedia also repeats names and words, and romaji repeats names and grammatical endings. I found no reliable vocabulary-size cue from the token subspace. I therefore lean only slightly towards Python, with English next; the corpus call is weak.

corpus: python
order: as is
probabilities: python=0.40 romaji=0.27 english=0.33; as is=0.80 shuffled=0.20