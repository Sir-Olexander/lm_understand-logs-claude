I'm noticing the OV eigenvalues for layers 1 through 3 come out strongly negative, around -0.95 to -0.99, even with the LN weights folded into V. That would mean these heads write back the negative of what they read along the same residual directions, which seems odd enough that I want to double-check it.

The eigenvalues being negative for every head in layers 1 through 3 looks systematic rather than random, since randomly initialized heads should give realfrac near zero. A shared effect like a large common residual direction being subtracted could explain it, though I'm not sure yet.

I'm wondering whether the consistent negativity could come from shuffled text, where heads might be averaging over characters already seen in the line. Whether seeing a character makes another of the same kind more or less likely depends on whether the line's character mix is fixed, and I haven't settled that yet.

I'm checking whether the QK side matters here: its eigenvalues are tiny, around 0.0 to 0.3, and I'm recomputing them from the scaled query matrix since the query norms are fairly large. The OV eigenvalues act on residual directions rather than token identities, so the negativity could come from heads cancelling a large shared direction like a bias or positional component. The singular values for layer 1 are fairly flat, so the map isn't low rank.

I'm checking whether the eigenvalues of the QK product come out small only because Q and K are close to orthogonal. Singular values and the symmetric part of that product might show more than the raw eigenvalues do. I also want to look at layer 0's input LayerNorm weights, where the median is around 0.18 and most dimensions are heavily downweighted.

Now I'm comparing the layer 0 heads: H4 and H0 have small query and key norms, so their attention should be close to uniform over earlier positions, and H4's output is large and low-rank with a positive realfrac. H1 and H5 have noticeably larger query and key norms, so they should attend more selectively.

Checking whether the initial weight scale changes the picture: at init, Q has a Frobenius norm near 0.6 once the input LayerNorm weights are applied, so the observed 1.9 is larger than that. Since most of the LayerNorm weights are small, what matters is which input dimensions each head actually reads from.

I'm now looking at the LayerNorm biases across layers 1 and 2, where dims 123, 21, 51, 307, 125, and 35 keep showing up, which could mark them as massive-activation or otherwise important dimensions. I'm also noting that most of the layer 0 input LayerNorm weights are small, around 0.18, with only a few dimensions near 1.

I'm weighing whether the layer 0 heads should be read as uniform or selective attention. The first head, with tiny query and key norms, should spread its attention broadly over earlier positions and produce a large, low-rank output. Before settling on anything, I want to check whether the input direction the heads read from, or the positional structure, explains that pattern.

I'm comparing the layer 0 heads' attention patterns, and checking the QK eigenvalues for L0H5 and L0H1, where the angles look rotation-like. I'll look at the singular values and the symmetric and antisymmetric parts of that product, since the raw eigenvalues are small. I also want to see whether the positional embedding shows up in that structure.

I'm going to compute an SVD of the query-key product for each layer-0 head and look at the principal angles between the query and key row spaces. I'm also thinking about whether the positional embeddings show up as a shared subspace that both query and key read from.

I'm now thinking about how to separate token and position directions in the layer-0 inputs, and whether heads that attend by position read a positional subspace on both query and key side. The plan is to check K-composition from layer-0 heads into layer-1 keys, to see whether a previous-token mechanism shows up that way.

I'm now looking at how the composed bilinear form from layer 1's queries and keys through layer 0's OV circuit could detect a token repeated from the previous position, which is the induction pattern. The check would be whether that matrix restricted to the token subspace has positive diagonal entries or positive eigenvalues, compared against what a random-like matrix would give.

I'm checking the layer-1 query and layer-0 value composition: M = Q1ᵀK1·O0·V0, including the LayerNorm scalings, to see whether its dominant eigenvalues are positive and real. If they are, the head pair matches whatever the value circuit reads from the source position, which is the signature I'd expect from an induction-style mechanism.

I'm now working out whether the value-side composition M = Q1ᵀK1·O0·V0, with the LayerNorm scalings included, has positive real dominant eigenvalues, since that would indicate an induction-style match. I'm also weighing the realfrac of the layer-1 to layer-3 OV eigenvalues, which look strongly negative, and checking whether the scaling of V explains that.

I'm wondering whether the consistently negative OV eigenvalues across layers 1 to 3 could come from these heads subtracting a large shared residual direction, possibly one tied to the massive-activation dimensions. I'm also considering whether training dynamics could push the output weights toward the negative of the value weights.

I'm weighing whether a negative OV circuit could mean these heads write back against the residual directions they read, perhaps as the model subtracting a large shared component. Whether the output-side mapping matches the embedding directions, as in an untied model, is still an open question I haven't settled.

I'm wondering whether a negative OV circuit is typical in ordinary transformers, since I recall GPT-2 copying heads having positive eigenvalues in token space but am unsure about residual-to-residual ones. The exchangeability argument suggests seen characters should positively predict repeats, so the copying direction would be expected to be positive.

I'm going to check whether the most negative OV eigenvalues line up with the large-magnitude standard-basis dimensions, and I'll also compute the per-head SVD of the query-key product to compare the top singular directions.

I'm normalizing the composition matrices by their Frobenius norms so the magnitudes are comparable across head pairs, and I'm also checking the Q-composition side. Next I want to see whether the OV eigenvectors line up with the large standard-basis dimensions.

