The L2 and L3 heads mostly read layer-1 and MLP outputs, so these L0-derived subspaces barely capture them, and they look roughly like baseline except for the Fo keys. I need a more global approach for the later layers, since they read features built by the L0 MLP and the L1 outputs. That part is hard.

I'm thinking that the simpler reading of the question might be what's intended: prior-token and induction-style heads in early layers, broader heads later. But I can't run forward passes, and building approximate inputs to test would be off-limits, so any corpus guess would have to come from structural features of the weights, like how strongly they encode line-level position and indentation.

I'm weighing a few structural signals that could hint at the corpus without running forward passes. Python code is the most predictable of the candidates, and romaji sits a bit above it, so the output scale of the final layers might carry some trace of confidence, though I'd need reference models to compare norms meaningfully. I'm also keeping the attention-sink pattern in mind, since it's universal and won't distinguish corpora.

I'm working out what a shuffled model's heads would likely do: a broad head restricted to the current line, detecting line starts via newline attention, and possibly heads that copy token identity from nearly uniform attention, such as L3H4.

I'm fairly sure the model is trained on real text rather than shuffled, since L0H0's query-key pattern shows a recency or relative-offset bias on top of a positive diagonal. I want to check the later layers for corpus-specific signals too, and I'm now thinking about which corpus would make an attend-to-a-different-character head useful, with Python indentation as the likely candidate.

I'm now thinking the head that avoids attending to its own token could be a "recent different character" head, picking up things like double letters in English or double consonants and long vowels in romaji. The sink bias on top of that would make these heads mostly track the most recent character that differs from the current one. I'm also wondering whether the same head could help count indentation by measuring distance back to the newline, though I haven't settled on a way to pin down the token identities from the logit structure.

Now I'm checking the singular values of L0H3's OV map from the token embedding, which show about 18 strong directions and suggest the token set it handles is roughly the size of a 26-letter alphabet plus a couple of whitespace tokens.

I'm trying to pin down which tokens the L0H3 OV map is actually distinguishing, since the bigram structure depends on every token's identity. Rare letters like z, q, x, and j probably carry weaker singular values, so the count of strong directions could tell English from romaji.

I'm checking whether MLP0 neurons act as bigram detectors, combining the current token with the previous one through the L0H3 output. Since the MLP has no bias term, each neuron's pattern over the 28 token points is what I'd need to read off.

I'm now considering that the positional embeddings might carry a period-4 component, which would let the model track indentation in multiples of four. A head could then attend to the same column in the previous line and copy whether that position held a space, which would require line-start features to be available.

I'm checking whether the position-copying features from the L0 attention heads can let a later head line up the same column in the previous line, comparing each query's offset from its line start against the key's offset.

I'm checking whether the Fout features are token-dependent, since L0H5's key direction leans heavily on the token embedding even though the head itself is nearly uniform.

I'm looking at whether the newline token might carry a large embedding component, which would make it stand out in the L0H1, H2 and H4 scores. For those heads, the query is positional and the key mixes token and position. I'm also thinking the Fout feature acts like a sink: with a periodic positional signal, its average shrinks toward zero after the first few positions, so it's only large near the start of the sequence.

Checking whether L0H5's key preference is token-driven or just a mild modulation, since its query-key scores are small. I'm going with the usual massive-activation explanation for the sink here: a few residual dimensions set up early, which points toward a sink and doesn't say much about the corpus.

Now I'm wondering whether the anti-same-token pattern in L1 points to Python-style indentation, where long runs of spaces would make a head skip past them to find the last non-space character.

I'm checking whether the anti-same-token pattern across many tokens, not just spaces, points toward Python's long runs of repeated indentation characters.

I'm checking whether the L1 heads' value vectors carry any positional information, since copying the newline's position would be how they'd measure run length. The numbers show they don't: the value weights are dominated by token and H3o components, with almost nothing on the positional side, so the heads don't seem to copy position.

I'm weighing whether the L0H1/2/4 key directions respond to the newline token, since a token-dependent key would let those heads measure distance back to the last newline. I'm also considering that the same newline signal could show up in the L0H5 and L1 patterns, though I haven't checked the L3 logit directions against it yet.

Next I'm testing whether the singular spectrum of W_fc.0, restricted to the token subspace T, shows a clear cluster that would reveal how many distinct tokens the model separates. I'd compare that against spectra on other subspaces, though a raw spectrum like that may not give a clean token count. I'm also considering whether composing the L0H3 OV map with a later head's value map could expose a rank structure, though that seems smooth and unlikely to reveal sharp boundaries.

Now I'm looking at the final layer's MLP output, which has to fit within the logit space. The singular values of W_proj.3 drop off gradually after the first 21 strong directions, so I need to find where they fall into the noise floor to estimate how many token dimensions the model actually uses.

Now I'm comparing the weight spectrum against the logit count: about 24 strong directions plus a handful of weaker pairs around 4.6 to 6.4 gives roughly 28, which fits English with its rare letters j, q, x, and z better than a 24-token romaji set.

Now I'm checking whether the logit-space count holds up another way, by looking at the union of the output-relevant subspaces across the layers.

Next I'm checking the principal angles between the top W_proj.3 directions and the L3H3 output directions, and also looking more carefully at the squared-singular-value spread of W_proj.3 to see whether the bulk writes to non-logit directions.

I'm checking whether the 4 weaker directions in the W_proj.3 spectrum could come from rare Latin-script words in the corpus, since that would explain why l, q, v, and x show up in the logit space despite being uncommon in romaji.

I'm weighing whether the weak directions could be l, q, v, and x in romaji too, since rare outputs like those might still get a logit direction strong enough to show up in the spectrum.

I'm going to check whether the top output direction of the final MLP, the one with a big gap after it, lines up with a vowel-versus-consonant split. I'm also looking at the L3H3 copy head's OV singular values against the top subspace of W_proj.3, since a copy head should span the full logit space, which would tell me roughly how many tokens are in play.

I'm looking at the spectrum of the L3H3 OV map projected onto the top 28 directions of W_proj.3, since that shows whether the copy head's output has any sharp cutoff inside the logit space.

