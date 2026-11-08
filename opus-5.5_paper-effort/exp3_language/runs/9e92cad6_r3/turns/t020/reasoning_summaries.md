The residual signal is concentrated in coordinates 21, 99, 58, and 39, with low participation ratios, which points to privileged massive-activation dimensions. For coordinate 21, I'm checking the weights: the MLP layer norms barely touch it, while the first attention layer's norm reads it strongly.

I'm noticing that the high-norm coordinates in layer 0's LayerNorm weights (180, 184, 162, 99, 211) line up with the ones the L0H5 input reads, so the input seems to live in the dimensions that layer 0 kept large. That makes me wonder whether these few coordinates carry a positional signal, something like a position ramp encoded in a privileged subset.

Next I'm thinking about how the position-0 embedding might drive the massive activation: if its Fin component is distinctive, Fout would be large there and decay over later positions, which would make it an attention sink. The Fout reads in the L1 keys would then mostly carry a positional bias rather than any segmentation signal.

I'm reconsidering the "segment" reading, since the heads' content must sit in the remaining components. L1 heads seem to suppress same-token content, with queries reading the head-3 output and the token signal, and keys and values drawing on the token signal. The clearest evidence still points to L0H3 as a sharp positional-offset head and L3H3 as an induction head writing to the logits. For the shuffled hypothesis, I have no positive evidence beyond the absence of more positional heads, and I'd still need a newline-detecting mechanism to identify the current line, which I haven't found. I'm now checking whether L0H3's offset might be attending to the previous token.

I'm checking whether L0H3 might simply attend to the previous token, since its positional offset would fit that. For the corpus question, I'm looking for more structure, and I'm considering whether the L1 heads that avoid same-token content would matter for something like Python indentation, where a space inside a run needs to skip other spaces and find the last non-space character.

I'm weighing whether the L1 heads that avoid same-token content could help with something like Python indentation, where a space has to skip other spaces to find the last non-space character. The space token looks like the most frequent one, which fits a corpus with a lot of indentation or word gaps, but I still need concrete token-level structure to tell the candidates apart. I'm considering whether the output-side direction with the largest singular value might map to a space.

I'm working through whether the output direction with the largest singular value maps to a bigram-level structure, and I'm considering the induction head L3H3 as a copy mechanism that maps each token to itself in the logits.

Working through the invariants question now: spectra of products of OV matrices aren't basis-invariant when the embeddings are non-orthonormal, so the singular values in residual space may be distorted. I'm hoping the Gram structure is roughly isotropic, and I'm also wondering whether the number of distinct copy directions in L0H3's OV could tell me something about how many tokens the model separates.

I'm trying to pin down what the head's copying actually encodes: for L0H3's OV, I'm asking whether the token identities separate cleanly, and whether the spectrum reflects how much downstream computation depends on each previous character.

I'm looking at how the token embedding is spread across dimensions, and noting that the raw token component should span at most about 28 directions, while position may add more. The spectrum from the stacked run is smooth with no clear cutoff, which I'm attributing to initialization noise rather than real structure.

I'm checking the layer-0 input space, which spans many dimensions, so it doesn't reveal much on its own. Next I'm looking at whether the large layer-0 norm weights at coordinates like 180, 184, 39, and 99 could mark where positional encodings concentrate, though that seems limited. I'm also considering whether indentation runs in Python might tell me something about the corpus.

I'm working through how normalization changes the text: non-letters become spaces and runs collapse, so Python code turns into mostly keywords and identifiers, with indentation runs kept at the start of lines. I'm wondering whether lines made only of symbols end up as empty lines after trailing spaces are stripped.

I'm working out why the Python case looks distinctive: with punctuation collapsed to spaces, lines are mostly identifiers with indentation runs, so tracking how far into a line we are is a key cue. Heads that copy positional offsets from the last newline would fit that, which would make Python a good candidate for the corpus. I still need to check that against the massive-activation dimensions (21, 99, 58, 39) that look like they're carrying the attention-sink signal.

I'm checking whether L0H5's output is really a sink mechanism: it would be a uniform average over the prefix, so it would track position rather than token identity, and the huge value at position 0 would be the telltale sign.

I'm sorting out the sink interpretation: the large Fout at position 0 would decay with position, so it wouldn't tell the corpora apart. That leaves the previous-token and copy heads as the real signal for ordering, and I'm wondering what they say about the corpus.

I'm checking how strong induction might be here: the only clear induction head is L3H3 in the final layer. In Python, I'd expect early induction from all the repeated identifiers, but Wikipedia-style English would show weaker induction, since repeated names within 128 characters are less common.

I'm weighing the L1 heads that avoid same-token content as a possible signal for Python indentation, and I'm considering whether L3H3 is the main head writing to the logits. I want to look at the L1 and L2 head structure more carefully, using the token, previous-token, output, and position subspaces, to see whether position within a line is what they track.

I'm checking whether the L0 heads at H1, H2, and H4 attend to newline or space tokens to track distance, since their keys depend on both token identity and position. Their scores come from position-dominated queries and mixed token and position keys, so they read as positional heads with some token modulation.

I'm checking whether the heads' key directions point at a specific token like newline or space, or whether they blend token and position. I'm also turning to the top logit-space direction and which heads and MLPs feed into it, then looking at L3H3's copy structure in the logit subspace to see whether the copied tokens have similar singular values.

I'm thinking about the neurons in the first-layer MLP as possible token detectors, and whether projecting them onto the subspaces of the previous-token head and the copy head could reveal clusters. I might try k-means on the normalized projections to find a set of token directions, though I'm not sure I can count distinct tokens this way.

I'm looking at the output neurons of the third-layer MLP, where each neuron writes into the logit subspace, and wondering whether they can reveal token structure without a known final layer norm.

I'm wondering whether the output neurons of the last MLP layer, projected onto the logit subspace, could be sparse in some basis, since each might promote a small set of tokens. If I can find that basis with ICA or a sparse dictionary, the atoms might line up with the unembedding directions and give me a token count.

I'm weighing whether L0H3's attention, with its offset pattern and the positional embedding, can separate the corpora. Its positional structure might show whether the heads track distance within a line, which would be the clearest mark of Python indentation.

I'm testing whether the L0 positional embeddings behave like a helix, where each step rotates the position vector by a fixed angle. If so, an attention offset of one would show up as roughly 45-degree eigen-angles in the L0H3 pattern, which would fit the periods I'm seeing.

I'm estimating the rotation angle of the L0H3 pattern against a range of offsets: if the head were a sharp previous-token head, the positional frequencies would be high, but a roughly 45° angle suggests a period near 8 rather than a sharp offset-one mechanism.

I'm checking whether L0H3 is a true previous-token head: with a period near 8, offset 1 wins only modestly over offset 0 and 2, but a steep softmax could still make it sharp. Since the induction composition with L3H3 points that way, I'm leaning toward labeling it "previous" for now.

I'm wondering whether the L0H0 head, with its positive real eigenvalues and decaying OV values, is really a self/recency head, or whether it mixes in the previous token too. Before I settle the head labels, I want a sharper way to tell the corpora apart.

I'm considering whether the model might track something like bigram or space-run structure instead, since indentation runs in Python would require counting position since the last newline to predict whether the next character is a space or a letter. I'm not sure this separates the corpora, since spaces also appear in English, so I'm weighing whether runs of spaces alone would be a usable signal.

I'm thinking about whether a period-4 positional feature could let the model count distance from the last newline, since Python indentation comes in multiples of four spaces. That would mean the L0 heads that copy position carry this periodic signal, though I'm treating it as speculative for now.

I'm checking whether the first-layer heads' input directions line up with a rotation plane of L0H3's attention, which would make them a periodic position feature with a frequency I can read off from the angle.

