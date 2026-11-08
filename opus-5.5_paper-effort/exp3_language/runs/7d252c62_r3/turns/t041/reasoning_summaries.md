I'm noting that uA gets copied only weakly by OV5 and OV1, and induction heads barely match it. That makes me wonder whether it's partly positional, or tied to a token the model doesn't copy, like a newline or space. Another possibility is that it's the layer-norm magnitude axis.

I'm weighing whether the remaining budget should go toward a different angle on the corpus. Indentation seems like the key feature here: in Python, a space at a line start is usually followed by another space, while in English or romaji double spaces are rare. That would mean the model has to track whether a space is indentation, and counting that may need longer context than the simple copying heads provide.

Now I'm weighing whether a counting-style head could track indentation-density in Python text, since that's a whitespace signal the simple copying heads don't capture. The romaji angle is also appealing: after a consonant, the vowel is nearly determined, so the useful dependencies sit at the syllable level, and I'm looking at how the relative spacing between syllables plays into that.

I'm going back over how the layer-0 embeddings might be reconstructed, since the LN weights seem to separate token content from positional signal. The correlations of the diagonal terms with the LN weights hint at which dimensions carry token identity, though I don't have a clear route from there yet. The biases on the layer norms look like constant offsets that won't tell me much, so I'm setting them aside.

I'm turning to the top MLP0 output direction, which has the largest singular value, and wondering whether layer-1 queries read it as a newline or space detector. The word-initial letter after a space depends on the previous word, which is true across corpora. For Python, the next newline might depend on syntax, and the most useful context may be only a few lines back.

I'm checking whether the L1 heads' strong negative q-k correlation on the same token reflects a deliberate avoidance of repeating the current token, since that would make the previous-position heads more informative.

I'm wondering whether uA is really about spaces, since prev-token heads would then have to handle indentation runs in Python. If so, copying space information could matter a lot for predicting indentation, even though the copying heads currently show only weak uA signal.

I'm considering whether uA is just a token-centroid direction: if so, a positional head keying on it would apply roughly uniform attention across tokens and carry little content. That would explain its weak OV copying and low MLP0 readout, so I'm treating it as possibly uninformative. Next I want to check uB, which L0H3 copies normally.

I'm checking whether uB is a real token direction, since L0H3 seems to track the distance back to the last uB-type token and L3H5 matches it above average. I can't yet tell whether it marks spaces or newlines, and I don't think I can pin down the corpus from the token identity alone.

I'm trying to work out whether the positional structure of Python text, with lines starting at column 0 and indentation in multiples of four, could be what these heads are picking up on. I'm also weighing whether induction matters much for character-level English versus Python. Before settling on a corpus guess, I want to look at the remaining unexplained heads, and maybe check the principal directions of the value weights for heads 1 and 5.

I'm comparing the principal angles between the raw value-projection row spaces of L0H1 and L0H5, looking for shared directions with cosine above 0.9 that would count as token dimensions. I'm planning to use unscaled rows, since the noise level is around 0.4 in singular value.

