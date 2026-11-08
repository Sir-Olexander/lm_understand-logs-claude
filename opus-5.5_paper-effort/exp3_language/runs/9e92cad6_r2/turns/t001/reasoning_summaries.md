I'm starting by looking at the weights to see what I can learn from them. The main question is whether the training text was shuffled, and without embeddings that's hard to pin down. Still, I can study the attention structure, and in shuffled text attention to the previous character would tell me little beyond the characters within a line.

I'm now weighing whether the shuffling would leave any usable signal in the attention, since the characters within a line keep their multiset but lose their order. Induction-style heads should be weak here, while heads that spread over the whole line might still help with the line's character distribution.

I'm weighing how to tell head types apart without positional embeddings, and wondering whether the weight matrices' structure can reveal a positional subspace, since the token embeddings only span a small slice of the residual space.

I'm checking whether Adam's per-element normalization breaks any subspace structure in the layer-0 weights, which makes the weight-decay argument shaky. The input space to layer 0 should be at most about 156 dimensions, so the right singular vectors of W_q might expose it. I'm also considering how the corpus differences, like Python's heavy indentation, could show up in the attention patterns.

I'm checking which features set the corpora apart: Python's indentation and frequent keywords on one side, and romaji's vowel-heavy, consonant-vowel rhythm with missing letters like l, q, and x on the other.

I'm sorting out how to get token identity back from the layer-0 weights, since the output projection isn't given to me. I'm also listing what the weights alone can reveal: the QK and OV matrices' ranks and norms, and whether the eigenvalues of W_O W_V show copying-style heads.

I'm weighing how the OV circuit's eigenvalues might show copying behavior, and whether composition scores between the heads could reveal an induction circuit, with a previous-token head feeding an induction head's keys.

I'm now thinking about locating the positional subspace from the weights alone, since the position embeddings are smooth and might be recoverable from structure in the QK matrices, even though I don't have them directly.

I'm working out how the shuffled-versus-ordered distinction would show up in the model: within a line the characters are a random permutation, so the best predictor leans on the line's character counts so far and the position in the line, not on the order of the characters seen.

I'm now working out how the shuffling would show up in the attention: within a line the characters are exchangeable, so broad attention over everything since the last newline would be the sensible pattern. To know where a line starts, a head would need to pick out the newline tokens, and that's still open.

I'm now thinking about how to separate the two cases from weights alone: a shuffled corpus should leave attention mostly content-based and roughly uniform over the current line, while the as-is text would need previous-token and induction heads, which I could look for through composition scores.

I'm considering whether the layer-0 weights could reveal the positional subspace, since a head that tracks the last newline would need a recency signal from the position embeddings.

I'm treating the layer-0 weight matrices as the only clues I have, since the position and token embeddings aren't given. Adam's per-coordinate updates probably smear the signal across directions outside the input subspace, so I'd look for the concentrated part of the spectrum and compare the layer-0 singular values against deeper layers.

I'm asking whether shuffling would even show up in the weights: a shuffled corpus should leave attention broad, while the as-is text would rely on previous-token heads. Without the positional embeddings I can't directly test the previous-token pattern in layer-0 QK, so I'm looking at whether the positional part of the QK matrices can be recovered at all.

I'm wondering whether the position embeddings could be reconstructed from the weights, and whether the attention or MLP weights might carry letter-frequency information that would reveal the corpus. Without the output projection, I can't read the token directions off directly, so I'm checking what the layer-0 OV circuits and any clustering of their directions can still show.

I'm weighing whether a spectral read of the layer-0 weights could separate the corpora, and whether the count of effective token directions might differ: Python uses all 28 characters, while romaji leaves out letters like l, q, and x.

Now I'm wondering whether the shorter token subspace shows up in the layer-0 QK spectrum, since letters like l, q, and x would barely train. I'm also turning to Python's indentation, where the model has to count leading spaces to know when the indent ends.

I'm weighing whether the position embeddings could be pulled out of the weights, since a previous-token head would show up as a positional pattern in the layer-0 query-key matrices. For composition, I'm checking whether a previous-token head feeds into a later induction head's keys.

I'm weighing whether the OV eigenvalues could show copying in residual space, since the token embeddings and unembedding aren't available to anchor the read and write directions.

I'm considering what would distinguish shuffled text in the weights. A head that attends broadly over the context would have a QK matrix close to zero, since its logits would be roughly constant, but attending evenly over only the current line needs a way to tell lines apart.

I'm weighing whether shuffling would show up as several heads with near-zero QK norms, since uniform attention over the window is what a bag-of-characters model would need. Sharp attention, like a previous-token head, would instead need large QK norms. So I'm planning to compare per-head QK norms and sharpness, and I'll keep the layer-norm bias term in mind as a query-independent contribution.

I'm checking the weight scale against the random init: decay over training should have kept most of the initial noise, so I'd expect the trained structure to stand out only in the larger singular values. Next I'll print shapes, norms, and per-head QK and OV singular values, along with the layer-norm weight statistics.

