Two attention heads in layer 2 look like they're doing induction: one matches the query's current token against the key's previous token, the other matches the query's previous token against the key's prev-prev token, each with a strong score around 0.75 to 0.8. The offsets line up with a two-token context, and the same-offset matches on the other pair are also in the mid-0.6 range.

Now I'm checking the layer 1 heads, where the same-token query-key match comes out strongly negative, so they're steering away from attending to the current token itself. The remaining alignments there look weaker and I'm still working out what the last one is pointing at.

Now I'm weighing whether the L1 heads are leaning toward the immediately preceding token, since they suppress same-token attention but still show mild preference for the position just before. That mix looks unclear, so I'll check the offsets again against the L0 heads and see whether the L2H2 composition supports swapping them.

I'm checking the layer 0 offsets against the layer 1 and layer 2 patterns: a shifted induction pattern would show up as [qT,kprev1] and [qprev1,kprev5], but the L2H1 values there are near zero, so the offset-1 reading for that head holds. The offset-2 reading for the other head is the only consistent induction story, and L0H1 would then be a broader local head covering a few characters back.

I'm thinking about how to tell python, romaji and English apart from the attention structure. One route is the effective rank of the token-matching bilinear forms in L2H1, L3H1 and L3H5: if the spectrum drops off at a different point for each corpus, that could separate them. I'd need a larger token dimension to check that.

Now I'm trying to get at the token structure more directly: taking the symmetric part of the query-key matrix for the induction heads to count the positive eigenvalues above the noise floor. I also want to check whether the prev-token head's OV rank bounds how many distinct tokens the model can tell apart.

Now I'm looking at the singular value spectra of the query-key matrix, which decay gradually without an obvious cutoff, so they don't reveal how many tokens the model distinguishes. Next I'm considering corpus-specific structure: romaji almost always ends syllables in vowels, which limits consonant clusters, and in Python, indentation produces runs of spaces that appear only at line starts.

Now I'm weighing the corpus question through frequency: whether the same identifiers and tokens repeat densely enough in Python to make the induction heads dominant, and whether rare tokens get treated differently by the LayerNorm.

Now I'm thinking about whether the token embeddings cluster into vowels and consonants, since romaji's consonant-vowel alternation would make that split the most important feature for the prev-token head's OV to pick up.

Now I'm wondering whether a head can be found that keys on newline tokens, since in Python the line-start position carries indentation information that a newline-tracking head could pick up, while in English and romaji paragraph breaks are far apart and rarely inside the window.

Now I'm working out which token X the L0H3 head is keyed on: if it tracks spaces, it measures distance since the last word boundary, but if it tracks newlines, it measures position within a line, which would matter mainly for Python.

I'm trying to tell whether the prev-token head's input directions track only the token embeddings, so counting how many directions it copies might give a rough number of distinct tokens the model separates. The positional part of the input complicates that count, so I'm working out how to subtract it.

Now I'm reading the singular values of the prev-token head's OV matrix, and the tail past about the 21st value drops off slowly, so I'm wondering what level counts as noise and whether random initialization still contributes to it.

I'm estimating the noise floor for the OV matrix: a random product of the O and V matrices with the layer-norm weight comes out around 0.006 to 0.012, so anything near 0.01 is just initialization. The tail singular values from about 0.13 to 0.35 sit well above that, which means the prev-token head's OV has real learned rank beyond the first 21 directions.

Now I'm looking at whether the prev-token head's OV also carries positional information, since the tail singular values might reflect the previous position's embedding leaking through the layer norm rather than learned token directions. I'm also considering whether L0H1 and L0H5 share a key-token preference, which would show up as matching query-position and key-token patterns.

Now I'm wondering whether the L1 heads consistently avoid attending to tokens identical to the current one, since that would be an unusual pattern across the whole layer and might point to a specific corpus property.

The next part is mostly me talking through a possible reason the layer-1 heads suppress same-token keys: maybe spaces, or some other frequent token, are the reason. I'm also weighing whether the layer-1 heads point at the previous token or the induction target, and whether the statistics of repeated letters in romaji or identifiers in Python could explain it, though I don't see a clear answer yet.

Now I'm working through the indentation case: after a newline, the model would need a head that matches the current line's leading spaces against the previous line's, which looks like a basic induction pattern on "newline plus spaces."

Now I'm checking whether the token-identity form in L3H5 has enough positive eigenvalues to estimate how many distinct tokens the model separates, and I'm looking at the symmetric part restricted to the token subspace.

Now I'm computing the symmetric part of the L3H5 query-key blocks and the L2H1 block over the 40-dimensional token subspace, counting the eigenvalues that stay clearly positive to estimate how many distinct tokens the model separates.

