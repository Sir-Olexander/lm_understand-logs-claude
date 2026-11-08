Layer 0 looks uneven: two heads (H0, H4) have almost no QK signal, two (H2, H3) are small, and H1 and H5 carry most of the weight, which tracks with the layer-0 norm gains being small in many dimensions. I should note that these QK eigenvalues leave out the LayerNorm bias terms, and the real attention pattern also depends on the actual embedding vectors. Moving to layer 1, every head has a dominant complex pair near ±73°, plus a number of negative eigenvalues.

Wondering whether the common ±73° pair across all layer 1 heads reflects shared positional structure, since a rotation at that angle would map onto a particular position offset under a sinusoidal encoding. Negative real fraction suggests these heads push attention away from similar tokens rather than toward them.

I'm thinking about how to recover the positional and token subspaces, since residual stream at layer 0 is token plus position. Next I want to check whether the layer-0 MLP acts as an extended embedding that carries token-dependent information.

Looking at how the LayerNorm weights split across dimensions, with ln1_w.0 and ln2_w.0 anticorrelated, I'm wondering whether position and token information occupy different dimensions, since Adam's per-dimension updates could favor a partly privileged basis.

Checking whether the shuffled-text case would reshape the head roles: with shuffled text, a head would attend broadly over the current line and nearby lines rather than to a fixed previous token.

Layer 0 shows most of its attention weight concentrated in H1 and H5, while H0 and H4 barely register. Now I'm turning to the layer-0 QK eigen-analysis to work out what the shuffled-versus-as-is comparison would reveal about head roles.

Trying to separate the positional subspace from the token subspace: the token embeddings span at most 28 dimensions, the positional ones up to 128, and the layer-0 residual is their sum. The random part of the layer-0 weights should be fairly small once scaled by the LayerNorm gains, so the learned structure should stand out in the scaled singular values.

I'm weighing whether induction-style behavior would show up as copying in the OV circuits, which means checking eigenvalues of the residual-to-residual OV product. Without the unembedding that's only a partial signal, since the output direction isn't measured against the vocabulary yet.

I'm working out how the query bias term contributes a per-key score, since it shifts the attention logits based on each key's normalized residual.

I'm weighing whether the shuffled-versus-as-is gap shows up in the global statistics: with shuffled lines, the loss floor is higher and deeper layers have less to exploit, so the layer-0 work would be mostly counting characters within the current line.

I'm now weighing whether layer 1's shared positional mechanism reflects line-start-relative structure, or whether those heads simply attend a few tokens back. Next I want to pull out the input-side principal directions of layer 0 and see how they overlap with layer 1's dominant QK directions.

Now I'm checking whether layer 1's heads read from the raw embedding subspace or from layer 0's outputs, since the residual at that point is the embedding plus the attention and MLP contributions from layer 0. I'm also counting how many of the large layer-0 LayerNorm gains line up with the columns of the query and key weights.

I'm weighing how to separate layer-0 heads that attend to the current line from ones that track the previous token, and whether layer 0 could detect newlines by content so a head can gather the whole current line.

I'm weighing whether the attention patterns sharpen or stay diffuse, since a positional query would pick out fixed offsets while a line-restricted one would spread weight over everything after the last newline. The OV side is harder to tell apart: a previous-token head and a broad head would both copy token identity, just with different averaging.

I'm noting that layer-0 heads H0 and H4 have almost no QK signal, so they probably attend broadly over the whole prefix rather than to a specific offset. I'm also checking whether the query bias could still give them a per-key preference that the bilinear eigenvalues miss.

I'm estimating the size of the bias-induced term by treating the normalized embedding as roughly norm 19.6, so the attention score scales with the QK eigenvalue times 384 divided by 8. That means even the small-eigenvalue heads could produce sharp scores if the embeddings happen to align with the relevant eigenvectors, so the near-zero QK signal doesn't rule out selective attention.

Checking the OV side next: I'll compute the output-value norm for each layer-0 head, since a head with near-zero QK signal but a sizable OV could still matter, and a tiny OV would mean the head is effectively dead.

I'm checking the output-value norm per layer-0 head as well, and also measuring how much the query bias term adds to each head's per-key scores.

