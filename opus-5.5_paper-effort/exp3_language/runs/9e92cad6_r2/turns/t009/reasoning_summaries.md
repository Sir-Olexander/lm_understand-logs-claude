Head 3 in layer 3 has QK eigenvalues that are almost all positive and real, with a trace ratio of 0.98, so it looks like a matching head that attends to keys resembling its query. That could be a copy or induction-style pattern, or simply same-line matching. Standard induction would split query and key into different subspaces, which wouldn't give this positive-definite structure unless the previous-token information is written into the same subspace as the current token.

I'm checking whether the positive, similarity-style QK structure could be a duplicate-token pattern, where the head attends to earlier occurrences of the current token. The negative OV trace makes me wonder whether that signals suppression, though the unembedding differs from the embedding, so I can't read output effects straight from it.

I'm going back over the setup: six models trained on three corpora in two orders, and the task is to tell them apart. The shuffled-order models probably lack a dedicated previous-token head, since the previous character there is just another draw from the same line's bag. Layer 0 head 3 looks like a sharp positional pattern, possibly attending to nearby or current positions, and I'm wondering whether it helps track how far the model is from the last newline.

Checking whether a local, recency-weighted pattern could let the model find the last newline, and whether a uniform average over the line would do the same job. Induction-style heads should show up in the as-is text across all three corpora, especially on repeated identifiers, names, and particles, so their absence would point toward the shuffled models. I'm also wondering whether the layer-0 group is really a previous-token mechanism, though a rank-2 output space seems too small to carry token identity.

I'm looking at the query side of layer 0 head 5, where the query vector sits mostly in the subspace that matters, but the key vector does not. So the attention logit depends on how the query's feature lines up with that 2D subspace, and that could mean the head is reading something like whether the current token is a space or newline.

Estimating the attention logit size for layer 0 head 5 from its small QK scale: with components of roughly 2.8 in each direction, the logit stays near 0.25, so the attention is close to uniform. Layer 0 head 1 has a larger scale, which could give logits around 1.2 or more if a dominant direction, such as newline versus non-newline, is large.

I'm planning to decompose the layer-0 input subspace via an SVD of the stacked query, key, value, and MLP weights, and compare its spectrum against a random baseline. The goal is to separate the token-driven directions from the positional ones, with the token part capped at roughly 27 dimensions after centering.

I'm now thinking about how to tell which input dimensions carry token information and which carry position, since the token part can span at most about 28 directions. Layer 0's heads read from a mix of both, and I'd like to separate them. Later positional heads, like the one at layer 3, show rotation-like positive QK eigenvalues, which could serve as a reference for what a positional direction looks like.

I'm weighing whether layer 3's near-positive QK patterns reflect local or same-line attention, comparing the symmetric positive structure on the positional subspace against the mixed token-and-position picture. Then I'm considering how to separate the token-driven and positional directions in layer 0, since the token part can only span about 28 dimensions.

I'm estimating how big the layer-0 token subspace is, roughly 22 letters plus space and newline, and whether unused letters keep near-random embeddings that would show up as a smaller learned dimension. A direct check would need the embeddings, so I'd have to infer it indirectly. I'm also wondering whether the newline-rare structure in long-paragraph text could be read from a 2D signal.

I'm weighing whether newlines are so frequent in the training text that line structure dominates the layer-0 heads, and how that differs for long-paragraph English and romaji versus short dialogue-heavy Japanese lines. A 128-character window often holds no newline in the long-paragraph corpora, which would make line identity matter less there, though it still matters whenever a line break does appear.

I'm considering that a bag-of-characters average over the prefix, carried in a rank-2 signal, could encode how many newlines have appeared so far, so keys just after a line break would stand out.

I'm checking whether the layer-0 head's query-key structure is positional or token-based, and whether it attends to newlines with a sharp key bias.

I'm wondering whether the layer-0 group's positional-looking directions could carry an averaged newline position instead of just position itself, since a head attending sharply to newlines and copying their position would produce something different from plain position.

Next I want a cleaner handle on the layer-0 group, so I'm splitting its input directions into a token-like set, taken from the OV inputs of heads 0 and 3, and a positional-like set, taken from the query and key inputs of head 3. I'll also check whether the MLP at layer 0 mostly reflects token identity.

I'm thinking about whether the layer-0 input directions split cleanly into token and positional parts, maybe by stacking the reading matrices and looking for a gap in the spectrum.

