I'm looking at the head L0H0: its queries depend mostly on position and its keys mostly on token identity, so its attention scores blend the two. Its value vectors carry positional information, so the head is effectively asking where the tokens of a particular class sit and returning their average position.

Newlines would be the thing this head latches onto. If the key preference points at newline tokens, it averages their positions across the context. With no newlines present, the attention would spread evenly, so the output would be roughly the midpoint of all prior positions. A space-preferring head would be less informative, since its average position says little about line structure.

Keys in earlier lines have smaller averaged positions, so a strong preference for large values would favor the current line and recent lines. That would produce a broad, uniform attention over the current line, which fits a bag-of-line model for shuffled text. Whether this holds depends on which token the key side selects: newlines would be sparse in English prose and common in Python, so the mechanism matters more there.

Now I'm checking whether a prev-token-style head could help in shuffled text, since the answer depends on exact subtraction of prefix character counts from the line's multiset, which averaged attention can only approximate.

I'm checking whether a prev-token head would help here, and it looks like it wouldn't: in a shuffled line the character after a newline is random, so the only usable signal is the bag of characters. Still, I'm wondering whether L0H2 might act more like a self-attending head, since its rotation angles are nonzero and include a π component, which self-attention alone wouldn't produce.

I'm considering whether a sharp relative-position head could help with shuffled text, and it doesn't look like it, since a fixed offset gives no useful signal when the line order is random. That makes L0H2 look more like a previous-token head, which points toward the as-is model. The L3H1 circuit would also be unhelpful for shuffled text, because matching a token to its predecessor carries no information there.

Now I'm checking which subspace the top positive eigenvectors of L3H1's OV-composed matrix fall in, token or positional, since a positional match would be useless for causal attention while a token match would indicate induction.

