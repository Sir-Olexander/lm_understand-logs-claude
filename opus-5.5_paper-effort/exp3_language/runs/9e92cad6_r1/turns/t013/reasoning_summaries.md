I'm checking the L3H3 induction eigenvalues: most are near-real and positive for the top twelve, so they look like they match, with roughly twelve significant before the drop. Next up is L3H4's OV, which has eight large singular values before they taper off, while its QK looks weak. Its value vectors seem to read from position O0.

Still looking at L3H4's QK, which reads from O0 and O5. Now I'm turning to the corpus, thinking about token-level signals I might infer from weight relationships alone, starting with the previous-token head's OV spectrum.

I'm now weighing whether the token embeddings have a dominant vowel-versus-consonant direction, which would show up as a strong singular direction in the previous-token head's OV or in MLP0. The L0H3 value-input spectrum looks flat, though, so there's no obvious dominant direction there. I'm also wondering whether the positional heads in layer 0 are tracking position since the start of a line, given how newline-heavy Python is, but the key directions I've checked only partly overlap the token and position subspaces, so that's hard to pin down from weights alone. An alternative is to recover the token embeddings from MLP0's input, since without the actual points I can't evaluate those quadratic and linear head functions directly.

Now I'm looking at the layer 1 heads, and they all appear to share nearly the same attention structure. I'm also comparing what those patterns would look like for romaji, English, and Python, with Python likely showing broader long-range heads and newline-driven behavior, English showing word-level and topic-level heads, and romaji being highly regular.

I'm working out whether the outlier dimensions act like a position phase: if they encode something like cos and sin of position, a 70° rotation in that plane would correspond to attending to positions at a fixed offset. That would give the layer 1 heads a relative-position mechanism through the outlier subspace, though I'd still need to confirm it.

I'm weighing whether the position embeddings carry anything like a phase, and I'm checking L0H5's output to see if it can tell me that. Without the actual position vectors I can't evaluate the OV map, so that stays unresolved. The heads L0H1, L0H2, and L0H4 look like they copy the position of particular tokens, most likely newlines, which would make them useful for tracking offset within a line, especially in Python.

I'm trying to estimate how many tokens are actually "live" by looking at the dimension of the token embedding subspace. Dead tokens keep small, near-initialization vectors that the downstream weights never learned to use, so they shouldn't show up as learned directions.

I'm working out how many token directions are actually learned, by checking the rank of L3H3's OV, since it should roughly match the number of tokens it copies. Its singular values taper gradually from about 10.8 down through the 6s, so there's no clean cutoff yet.

I'm checking the singular values of W_o3,3 W_v3,3 to see whether they keep decaying without a clean cutoff, which would mean L3H3's OV doesn't reveal a token count directly. I'm now trying a different route: finding the token subspace as the part of the residual stream that layer-0 value heads read but the position signal doesn't, starting with L0H3's value map.

I'm checking whether the value spectrum of L0H3 really spans 40+ directions, since the singular values stay above the random bulk well past index 40. If so, this head reads far more than a clean token subspace would suggest, so I'll need another way to count the live tokens.

I'm wondering whether the vocabulary-count approach is too noisy, since W_v's gradient mixes in position components as well as the token subspace. Maybe the rank of the value map would be closer to the 28 tokens plus some position dimensions. I'll try the output side instead, starting with the logit-lens directions from MLP3's projection.

Now I'm weighing the output side: the W_out spectrum has about 21 large singular values before tapering, and since the final layer norm scales by the whole residual, writes outside the logit row space still shift the effective temperature.

I'm comparing the logit-relevant dimensions against the spectrum of the final MLP's output projection. Romaji would need about 24 live token directions, while English needs slightly more, so the extra singular values at 6 and 4.6 might correspond to rarer tokens.

I'm counting the outlier-dimension spectrum against the output side now, and the gaps after the top twenty-one singular values look about as large as the one at eleven-to-nine, so the split between twenty-four and twenty-eight live token directions stays ambiguous. I'm going to compare the output subspaces of the layer-3 heads against the final MLP's projection to see which count they support.

I'm going to compute an SVD over the stacked layer-3 writers, both the W_proj.3 matrix and each L3 head's OV, to see whether their combined output directions concentrate in roughly 28 logit dimensions plus a few norm-control ones. Counting the dimensions that survive that stack should help settle the token-count question.

I'm going to compute the SVD of W_proj.3 and check, for each layer-3 head's OV output, how much of its energy falls inside the top-k left singular subspace for k between 20 and 40, to see whether the energy saturates near 24 or 28.

