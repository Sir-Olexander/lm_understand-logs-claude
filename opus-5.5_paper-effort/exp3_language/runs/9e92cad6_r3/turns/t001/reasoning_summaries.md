Let me look at the model's weights first. My plan is to infer the structure without embeddings. For shuffled text, the order within a line is random, so attention to the previous character would not help much, and the model would likely rely on broad attention over the characters in the line along with position and line length.

I'm working out how the layer-0 weights can reveal the input space, since the trained parts of the attention and MLP matrices should line up with the token and position inputs, with weight decay leaving some of the random initialization behind.

Next I'm looking at how the learned part of the layer-0 weights could be separated from the random initialization noise, since the singular values above the random bulk should point to the input subspace. The residual stream starts as 28 token vectors plus 128 position vectors, and I'm wondering whether the positional part spans only a low-dimensional, smooth subspace that could be picked out.

I'm weighing whether the token embeddings could be recovered from the rest of the model, since the output projection is unknown and that route stalls. Looking at the layer-0 attention QK matrix might show something about positional structure, but I don't have the position embeddings to compute it directly.

Now I'm checking the QK structure for induction heads, looking at whether the composed matrix from the previous-token head's OV circuit into the later layer's query-key matching shows a clear pattern in the token subspace.

I'm comparing the composition score of the induction head's QK matrix against its OV circuit, the ratio of Frobenius norms, and checking it against random baselines. Without the token embedding or output matrix, I'd only be able to approximate the token subspace from the learned parts of the layer-0 weights.

I'm considering whether the token space could be read off from the rare letters: romaji barely uses l, q, v, or x, so those few embeddings might sit near their initialization, though I don't have the embeddings to check that directly.

I'm now wondering whether the learned token subspace is closer to 24 dimensions than 28, since the rare romaji letters should barely move from initialization, though the position contribution and small gradients would make that hard to detect. I'm also considering what distinguishes Python here: many leading spaces and keywords like "self", "def", and "return", so position within a line matters more.

I'm weighing whether shuffled-text models would show broad, uniform attention within each line, since the order of characters carries no information, rather than the previous-token and induction patterns a normal corpus would produce. So the test is whether heads in this model attend broadly over the current line and back to earlier lines, with no induction-style copying.

I'm checking whether a previous-token head in layer 0 would show up in the QK matrix, expecting an antisymmetric, rotation-like part for positional attention, and comparing that against the rank-1 structure I'd expect from content-based heads.

I'm weighing whether the induction heads would show up as strong K-compositions between the layer-1 keys and the layer-0 OV outputs, with the queries reading the token subspace. Shuffled text should compose those layer-1 heads differently. Since I can't recover the token embeddings directly, I'm considering what in the attention structure would separate the corpora: Python should show long-range repetition of variable names, which means strong induction heads, along with heads that look back to the previous newline to track indentation.

I'm comparing how each corpus would show up in the attention structure: Python's short lines and frequent repetition should produce induction-style heads and indentation tracking, while long English paragraphs, once shuffled, would reduce to averaging over a bag of letters. Romaji literature falls somewhere in between, and I'm wondering whether its word-level repetition would leave a detectable mark.

Next I'm looking at the singular value spectra and norms of each head's weight matrices to see which ones carry structure rather than noise.

Next I'm checking the singular value spectra and norms of each head's QK and OV matrices, plus the layer-norm weight statistics, to see which parts carry real structure.

