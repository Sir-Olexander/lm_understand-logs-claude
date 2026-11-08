The spectrum falls off smoothly rather than dropping sharply, so there's no clean low-rank cut, and the embedding could use up to about 156 dimensions. The format hints that the tasks differ in corpus or ordering, so I need to work out which weight properties tell them apart. The main contrast is shuffled versus unshuffled lines: with shuffling, the model has to rely on the position within a line and an estimate of line length, plus a bag-of-words signal.

I'm weighing whether the heads that attend to newlines or a uniform bag of characters carry the positional signal, since the model can't get position directly from the query-key pattern. I'm also checking whether a layer-0 head with a large output-value weight is the main way context gets pooled.

I'm looking at which layer-1 and layer-2 heads might compose with the layer-0 heads, in particular whether a layer-1 head could pick up position within a line by averaging newline counts over the context. I'm also thinking about whether a recency-biased head that attends to newlines could recover the position of the last newline.

I'm checking whether the layer-1 and layer-2 heads all do something similar, since their spectra look alike with a large top singular value. I'm also looking at layer-0 head 5, whose value weights read only a two- or three-dimensional subspace while it attends uniformly over the context, and wondering whether its query-key scale is really negligible.

I'm checking whether the QK scores are actually negligible, since the layer-norm scale on the inputs could push attention logits well above what the raw matrix norms suggest. If the top singular direction lines up with a large component of the normalized embeddings, even a rank-1 head could attend sharply.

I'm estimating how large the query-key scores can get once the layer-norm scaling is included, since the normalized embeddings have norm around 19.6 and a rank-1 direction could still produce moderate attention. I also want to account for the query bias, which adds a key-dependent preference independent of the query. Next I want to see whether the token embedding geometry can be recovered from the layer-0 MLP inputs.

I'm now weighing the layer-3 heads, whose output weights are large, as the place where the answer is written out. I'm also turning to the corpus question: the clearest hint of the source would be indentation in Python, and telling that apart means checking whether heads that attend to newlines carry the count of leading spaces on each line.

I'm now weighing whether a layer-0 head's query-key matrix can reveal positional attention, assuming token and position subspaces are roughly orthogonal, so that the position-to-position block shows a previous-token pattern.

I'm weighing whether the asymmetry of the query-key matrix can reveal a previous-token pattern, since a positional head that shifts attention by one would be strongly non-symmetric. Content-matching heads should look closer to symmetric, though a rank-1 head that attends to newlines would also come out lopsided. I'm considering the scale of those ratios and what they would tell me about which heads are doing positional work.

I'm thinking about how the corpora might differ: Python code would have frequent newlines and indentation, while English Wikipedia lines are long paragraphs and Aozora romaji has a very regular consonant-vowel pattern. For the shuffled model, I'd expect it to lean on uniform bag-of-characters heads like L0H5 over the whole context, since long lines rarely put newlines in the window.

I'm checking whether a previous-token head could show up as an asymmetric query-key matrix. A shuffled model would also rely on the remaining-letters effect and on line-length estimates, which matter most for short Python lines.

I'm working out a composition score between the layer-0 previous-token head and the layer-1 heads, comparing how much a layer-1 head's query-key matrix reads from that head's output-value matrix. I'm also setting aside the query bias for now while I lay out the normalized formula.

I'm checking the composition baseline: random matrices of this shape would score roughly 0.05, so anything well above that would suggest a real connection between a layer-0 head's output and a later head's input.

