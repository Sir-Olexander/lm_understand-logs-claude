I'm starting by looking at the model's weights, since there are no embeddings to work from. The corpus and token order are unknown, so I'm considering whether attention heads could reveal them. Without the positional embedding, detecting previous-token heads from the QK matrices is hard to do directly.

I'm weighing whether the shuffled text would even let the model use order, since within a line the characters are random and only the bag of characters and line length carry signal. Induction-style copying wouldn't help there, so the useful cue might just be spotting where a newline falls.

I'm weighing whether previous-token heads could be detected from the weights alone, since the positional embeddings are missing and direct positional patterns aren't visible. One possibility is measuring K-composition between a layer-0 head's output and a layer-1 head's keys, which would signal an induction-style circuit; for shuffled text I wouldn't expect that structure at all.

I'm considering whether the positional embedding's structure could be inferred from the weights, since a previous-token head's QK matrix should map each position toward the one before it, giving a rotation-like, non-symmetric pattern with complex eigenvalues. Content-matching heads would look more symmetric by comparison. I'm also thinking the vocabulary statistics could help tell whether the corpus is Python, romaji, or English.

I'm now wondering whether the token embeddings could be approximated from the weights, perhaps through the layer norm parameters or the residual stream. I'm also considering that the initialization scale and how far the matrices moved during training might differ by corpus, since the step count is the same across runs.

I'm checking whether the token embeddings could be recovered from the weights, starting with the layer-0 projections that read the layer-normed input. The embedding matrix has rank at most 28, which is a useful constraint, and the gradient updates to those layer-0 weights are outer products involving the normed inputs, so their structure might still carry hints about the input distribution.

I'm working out how the update to the layer-0 weights is structured: each change is a sum of outer products with the normalized token-plus-position inputs, so its rows should fall in a limited subspace of roughly 155 dimensions. The initial random matrix is still present too, only shrunk a little by weight decay.

I'm checking how much the weights have drifted from initialization under weight decay, which leaves the random init mostly intact. The learned part should be low-rank, so the top singular directions of the layer-0 input projections might span the embedding space, though I can't recover the actual token embeddings from that alone.

I'm wondering whether the corpus could be told apart from the weights, since the attention patterns in the as-is text should show previous-token and induction heads. Python, English, and romaji all repeat names and words, so induction heads would show up in each, and indentation might be a further cue. I'm also thinking the MLP neurons in layer 0 might reveal something through their dead-neuron counts, though that's a weak signal. Given how hard this is, I'll settle on a best guess with rough probabilities rather than a certain answer.

I'm planning to compute basic norms for each weight matrix and compare them against the expected initialization scale. Next I'll look at singular values, then the QK and OV circuits per head, and check how strongly the layers compose with each other.

I'm asking whether the previous-token head's QK matrix would show a shift-like structure on the positional subspace, something like a rotation block pattern if the positional embedding is a smooth helix. The antisymmetric part of the bilinear form would then indicate whether the head prefers earlier or later positions.

I'm checking whether the antisymmetric part of the QK bilinear form reveals a positional-offset preference, since a shift-like pattern would show up there. Random initialization adds equal symmetric and antisymmetric noise, so I'd need to separate that out, perhaps by looking only at the top singular components of the learned part.

I'm checking whether the induction-style composition shows up in the weights: the layer-1 query reads the current token directly, the key reads the previous-token output of layer 0, so the combined map should be near-identity over tokens. Without the token embeddings I can't evaluate that exactly, so I'm looking at the composed matrix W_Q^T W_K W_OV for the same signal instead.

I'm checking whether the full 384x384 product of the induction QK and previous-token OV matrices has mostly positive eigenvalues, since that would point to copying behavior. The sum of eigenvalues over the sum of their magnitudes gives a rough score, though the noise from initialization will also be in there.

I'm checking whether the layer-1 induction composition shows up in the weights, by folding the layer-norm scales into the query and value-output matrices. I also need to account for the layer-0 output bias, which is just a constant offset.

I'm weighing whether newline-counting heads could let the model track position within a shuffled line, since that tells it when a line ends. Heads that attend back to the last newline would give that signal, and the character distribution from nearby lines would supply the rest.

I'm working out whether the shuffled text would show any copying, since repeated Python lines get reordered independently and so can't be copied. I expect the induction-style composition only appears in the as-is text, so the real test is whether the layer-1 circuit shows it in the weights.

I'm wondering whether the token labels could be recovered at all, since the embedding has only about 28 useful dimensions and I can't name the 28 tokens from the weights alone. Maybe the layer-0 MLP encodes bigram-like features that would hint at which characters go together, though that's speculative.

I'm wondering whether the token labels can be recovered at all, since the embedding has only about 28 useful dimensions and I can't name them from the weights alone. The output side might show a similar subspace, but without weight tying I can't link the two, so I'm looking at the corpus statistics instead, such as the spread of characters in normalized text.

I'm thinking about which letters show up in each corpus: romaji leaves out l, q, v and x, English uses everything with q, x and z rare, and Python uses all letters with x fairly common. The embeddings for unused tokens should stay near their random init, but I can't read them from the weights, so I'm wondering how rare tokens would appear in the blocks.

I'm weighing whether the way lines end could separate the three corpora, since Python lines are short with indentation while Wikipedia-style paragraphs run long. I'm also considering whether romaji's repeated syllables would leave a distinct local structure in the attention.

I'm thinking about the indentation cue: in Python, the number of leading spaces after a newline should be predictable from the previous line, so a head that copies from the previous line's start would carry that. The normalization kept leading spaces but collapsed other runs, so 4, 8, or 12 leading spaces would be the telltale pattern.

I'm now going to ignore the shuffled-indentation idea for the moment and just run basic statistics on the normalized text to see what I'm working with.

