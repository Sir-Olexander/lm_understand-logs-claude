I'm looking at OV eigenvalues in layers 1-3, which come out strongly negative, so the head may be writing the opposite of what it reads, a kind of suppression. I'm wondering whether, in the shuffled model, seeing a letter makes it less likely to appear again, but then I realize that in shuffled lines a repeated letter may actually raise its own probability.

I'm working through whether a negative OV eigenvalue in residual space means the head subtracts the attended token's representation, so it would write something like -tok_j into the current position. I'm not sure how that maps onto the output, since the link between the token and the unembedding is unclear. I also recall that copying heads are usually analyzed through W_U OV W_E rather than the residual-space OV, so a negative residual-space eigenvalue may just mean the head reduces what it reads.

I'm checking whether the strongly negative residual-space OV eigenvalues might come from heads attending broadly, so that the output is roughly the negated average of the context features, which would push each position toward whatever hasn't been seen yet. I'm not sure that would actually help in shuffled text, so I'm treating it as an open possibility.

Next I'm working out the shuffled-text predictor: given a prefix's letter multiset, the next character should be drawn from the remaining letters of the line, which I'd only be able to estimate from what's been seen so far. The seen letters are a sample of the line's composition, so their frequencies are a reasonable estimate, and that would push the head toward positive copying rather than suppression. The open question is how to predict the newline, which depends on knowing the line length L.

I'm looking at how the model would have to guess where a line ends, since the length isn't known and the hazard depends on line-length statistics and position. I'm also noticing that line-end handling could be a clue, like how spaces and word counts relate to length. Meanwhile, the negative OV eigenvalues in layers 1-3 stand out, since residual-space signs for information-moving heads should look more arbitrary than this.

I'm considering whether the strongly negative residual-space OV eigenvalues could mean a head that attends broadly and writes the negated average of its context, which would be a kind of copy suppression. I'm also thinking about induction heads, which attend to the position whose previous token matches the current one.

I'm weighing whether the untied OV direction could still be systematically anti-aligned with the token embeddings across all those heads, which would point to something consistent in how the network is organized. I'm also considering whether small models commonly show negative OV circuits like this, though I'm not certain of that pattern.

I'm now looking at the QK eigenvalues, where layer 1 and 2 heads show complex pairs with large imaginary parts, which would suggest a rotational, relative-position pattern. I'm also considering that the strongly negative real eigenvalues might mean the head avoids attending to similar positions, but I still need the positional subspace to tell whether it favors far-away positions.

I'm working out how to recover the positional and token subspaces directly from the weights. The layer 0 QK form has complex eigenvalue pairs with angles around 40 to 46 degrees, which looks like a rotation over position offsets, so this head probably attends to nearby earlier positions.

I'm going through the L0H3 eigenvalues and their phases to check whether this head really looks like a previous-token head. The other L0 heads, H1, H2, and H4, have rank-1 negative QK forms, and H5 looks uniform, so H3 is the main candidate. Even so, a previous-token head would be useless for shuffled text, which makes me wonder what L0 is actually doing there.

I'm considering whether, in shuffled lines, any local structure exists at all, and I conclude there isn't, since every permutation within a line is equally likely. The only real signals are the newline positions, the count of characters so far in the line, and the letter composition, both within the current line and in the previous one, which may predict the current line's length and content. That suggests a shuffled-trained model would lean on newline-attending heads and broad uniform heads rather than precise relative-position rotations.

I'm thinking about whether the positional embeddings are smooth in position, so I could recover the positional subspace and check whether L0H3's top QK eigenvectors live there. A head attending to the most recent newline would need a monotone recency term rather than a rotation, and rotations might also do that job. I'm still looking for a better way to isolate the token subspace, perhaps through the L0 MLP neurons, since the token embedding has only 28 vectors.

I'm working out how to recover the positional and token subspaces from the weights alone. The idea is to find where the layer 0 query, key, value, and MLP input directions overlap, then read the L0H3 rotation subspace as positional and the L0H5 OV input directions as token-based.

Now I'm checking whether the layer 0 key directions for H1 and H2 match up, since their rank-1 QK forms don't look aligned with their own query directions. If they share a common key direction, it might be a token feature like "newline" or "space." I'm also thinking about whether leading indentation after a newline could matter here.

Next I'm checking which weight signals could tell English, romaji, and Python apart, starting with how many output dimensions the W_proj.3 singular vectors use. If the strong components line up with the expected active token count for each language, that would be a usable fingerprint.

I'm checking the sign and magnitude of the layer 1-3 OV eigenvalues against the token-count signals in the output projection, since shuffled and python text would use the alphabet differently. Python likely exercises all 28 tokens fairly evenly, while English leaves j, q, x, and z rare, so the active-dimension count could help tell them apart.

Next I'm comparing the 21 large output singular values against the medium ones, to see whether the strong directions match frequent tokens and the weak ones match rare tokens. For English, that would mean j, q, x, and z hardly contribute, while romaji leaves out l, q, v, and x. The count alone doesn't settle it, so I'm thinking about checking for unused input embeddings instead.

I'm reconsidering whether the token embeddings can be inferred indirectly, since both ends of the circuit are unknown. One idea is to compare the heads with each other, for example the bag-of-letters heads in layers 1 through 3 and the uniform head L0H5, to see what information they share.

I'm turning to the L0H3 composition test: if it really is a previous-token head, its output should feed the L1/L2 induction-style heads through their keys. I want to compare that against the shuffled-order picture, where I'd expect no such previous-token structure.

I'm comparing the V-composition scores against baselines, either random orthogonal transforms or other heads, to see whether they stand out.

