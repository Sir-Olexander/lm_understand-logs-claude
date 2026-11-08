I'm checking the L0H2 QK eigenvalues, which come in complex conjugate pairs at various angles, consistent with a rotation-style relative-position structure. If the offset is δ=1, each angle should match a positional frequency, and I'm comparing the listed angles against that expectation.

I'm working out what the angles imply: a relative-position head with nonzero offset, most likely previous-token (δ=1), since δ=2 would give 2ω mod 2π. Without the frequencies I can't pin δ down directly, so I'm considering comparing against another positional head, L0H3, whose angle ratios would reveal the offset ratio.

I'm now checking which layer-0 heads write the top W_o directions, by measuring each head's contribution along those singular vectors. The question is whether this direction tracks something like the current line, which broad later heads would need in order to attend within a line.

I'm checking which layer-0 heads write the top W_o directions, looking at whether the broad near-uniform heads (L0H0, L0H1, L0H5) carry a strong single-direction output that could be something like a newline or position-in-line signal.

I'm looking at whether L0H0's key preference is query-independent: since the query is roughly constant, the score reduces to a key-only term, and I'm checking whether its top singular direction in M is rank-one enough to act like a fixed preference for certain tokens.

I'm trying to tell which tokens each broad layer-0 head prefers: if the score is mostly a fixed key preference, these heads might pick out newlines or spaces across the whole context, though that wouldn't favor recent positions unless position also plays a role. Without the token embeddings, I'm wondering whether I can approximate them from the residual stream to label what the top directions represent.

I'm working out how to identify the token-embedding subspace from layer 0, since the previous-token head H2's OV circuit maps those embeddings to its output.

I'm working out the L1 induction head's query-key score: its composition with the L0H2 previous-token OV circuit gives a bilinear form on the input space, and I'm setting that up as a candidate test for positive eigenvalues.

I'm checking whether the layer-1 induction head's query-key composition with the layer-0 previous-token OV circuit gives mostly positive real eigenvalues, since that would point to same-token matching. Without knowing the token subspace I can only run the eigen-analysis on the full 384-dimensional space, where matching circuits should show up as large positive real eigenvalues.

I'm now considering whether L0H2 could be a relative-position head with some other offset, but a sharp positional head doesn't seem useful for shuffled text. Since the Python indentation structure could still call for something like a fixed-column lookback, I'm leaning toward the as-is interpretation.

Next I'd like to confirm the induction circuit by checking whether the L1 query-key composition with L0H2's OV gives mostly positive real eigenvalues. For corpus identification, I'm weighing a few candidate signals: how many token dimensions actually get used, whether indentation-copying behavior shows up, and whether long lines with rare newlines fit the spectrum.

I'm weighing whether the spectrum of W_proj.3 settles the token count: the drop after the 23rd value is only about 20%, so it doesn't clearly point to 24 used tokens. I'm also considering whether rare characters could be suppressed through the bias term alone, leaving no extra residual dimensions for them.

I'm weighing the W_proj.3 spectrum against the two corpus guesses: a romaji-like set would leave about 23 strong directions, while English or Python would leave around 27. The count I'm seeing sits at roughly 28 or 29 above the bulk, which is hard to square with the romaji reading.

I'm weighing the medium singular values against which rare letters might account for them: in English, j, q, x, and z would each contribute a weaker unembedding direction, while Python would have fewer such letters because x, k, and v show up more often there. Romaji would leave l, q, v, and x nearly absent, so the count of weak directions would look different again.

I'm trying to pin down the token count and corpus from the singular values, but the medium-sized values don't settle it yet. I'll need a better handle, maybe by recovering the token geometry from layer 0 and comparing it against the unembedding.

I'm weighing whether the embedding norms from layer 0 can separate the rare letters from the common ones: rarely seen tokens should keep near-initialization norms, which might reveal a romaji-like set of about 24 used tokens. The positional part of the layer-0 reading weights is harder to separate out, since learned position embeddings are smooth and could occupy several effective dimensions.

I'm weighing whether the layer-0 embedding and positional subspaces can be separated by where the L0H2 rotation planes act, and I'm checking the ln1 weight distribution for a privileged coordinate basis that could help tell token dimensions from positional ones.

I'm weighing the Python-indentation angle: a model on Python would have to count leading spaces and copy the previous line's indentation, and its heads would attend to the same column one line back. English Wikipedia would instead show rare newlines and very long lines. Romaji text would show short dialogue lines and paragraph breaks, so I'm setting up the letter-inventory check through the layer-0 OV circuit to tell these apart.

I'm considering whether the induction query-key structure can count the distinct token identities: the eigenvalues restricted to the token subspace should be positive for each matched token, which would give roughly 27 for English or Python versus 23 for romaji. That count might be cleaner than the W_proj.3 spectrum, though noise could still blur it. I'm also thinking about romaji's strict consonant-vowel alternation, which would make the previous-character feature very informative, but I can't see that pattern directly from the weights yet.

I'm going to check the induction circuit next, scoring how strongly each later head's key composes with each layer-0 head's value output. Before that, the corpus question still needs a different angle than the singular values, so I'm considering whether head attention patterns, like a previous-line-start signal from indentation, could separate Python from English.

I'm computing K-composition scores between each later head's key and each layer-0 head's value-output path, using the Frobenius norms normalized by the individual matrices, with the layer-norm scaling included.

