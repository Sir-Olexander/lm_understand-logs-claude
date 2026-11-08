The key-side preference for L1 and L2 appears to sit mostly in the S channel from L0, so these heads get a fixed bias from those scalar features no matter what the query says, plus a query-dependent part that also reads S. Now I'm trying to estimate how big the dot product of that key direction with the normalized residual is, which depends on the norm of the normalized vector.

I'm working out the size of the score variation: the S coordinate of the normalized residual could be large enough that a few L0-computed scalar features end up setting the L1 attention pattern. That would make it a positional-style mechanism built from averages, and I'm wondering why the model would compute a scalar that way instead of using position embeddings directly.

I'm picturing the L1 key preference as a feature f_j that counts newlines among attended positions, so the heads would favor positions with a high value of it. I'm also considering whether the feature might instead be a first-position signal, where uniform attention to the start yields 1/(j+1), though position embeddings already give absolute position, which makes that less obviously needed. Then I'm wondering why all six L1 heads would rely on one such feature, since a line-based distinction between the current line and earlier lines would explain it.

I'm checking whether a feature that counts newlines, or measures distance since the last newline, could make the L1 key preference pick out the current line. A recency-weighted newline average would spike at each newline and decay after it, which would give a signal tied to line position. I'm weighing whether that fits better than a plain first-position signal.

I'm now asking whether the feature r could be a "space" indicator, so that f_j tracks distance since the last space and marks where the current word starts. That would matter for L1 heads that look for the start of the current word. To pin down r, I'm checking whether it also shows up in MLP0's input, and whether the sharp previous-token head L0H2 is the piece that separates as-is text from shuffled text.

I'm checking whether L0H2's trace of 0.29 really separates it from a content-based head: for a random 384×384 matrix the trace would only be around 0.05 of the Frobenius norm, so that value is significantly positive, though well short of what a pure same-token head would show.

I'm checking whether L0H2's QK is mostly positional, which would mean its query and key sides live in a subspace separate from the token-identity directions its OV reads, as expected for a previous-token head.

I'm checking whether L0H2's OV input space, the directions it reads tokens from, lines up with its QK subspaces, since a previous-token head should keep positional and token directions largely separate. I'm also weighing whether the symmetry of its positional QK would tell me if it attends to itself or to the previous position.

I'm checking whether L0H2 attends to the last newline instead of the previous character. If it did, its value vectors would carry the newline's positional embedding, which would fit a shuffled-text mechanism for tracking position within a line.

I'm checking whether L0H2 is a token-based head or a positional one: its top key direction reads the same r feature that heads H0, H1, and H4 average, and the question is whether its value side pulls from token directions or positional ones, which would separate a "copy the last newline's position" mechanism from a plain previous-token head.

I'm now going to compare the subspaces: the output-side directions of heads H0, H1, H4, and H5 against the top directions of L0H2's output and its query and key sides, to see how much they overlap.

