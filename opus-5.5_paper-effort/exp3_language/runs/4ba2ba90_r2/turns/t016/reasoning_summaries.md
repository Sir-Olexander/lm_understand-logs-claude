I'm looking at direction U0, which everyone writes to, with a fairly low ln2 weight there, so it's probably a massive/mean direction rather than a logit one. U29 and U30 are written by many heads, including the broad L0 ones, which points to positional features. I'm also checking where U24 comes from, with L3H5 contributing the largest share.

I'm checking which of the W_proj.3 singular directions carry actual logit information. U1–U22 are strongly written by the copy head, and U23–U27 look like a medium tier that could still belong to the logit set. Counting dimensions, the total lines up with either a romaji or an English vocabulary, which is why the partition is still ambiguous.

I'm weighing whether U24, which L3H5 writes into most heavily, belongs to the logit set or is just a broad attention-driven feature. Its level of writing is close to the weakly written U25–U27 group, so it's hard to place. The count still doesn't settle it, since 23 dimensions fit a romaji vocabulary and 27 fit an English one.

I'm wondering whether the alphabet can be read off the layer-0 embedding geometry instead, by looking at the stacked attention reading matrix outside the positional subspace. The OV spectrum there shows no clean gap, though, and I'm checking whether the stacked singular values around the 23–27 range show a break.

Next I'm checking whether the layer-0 MLP's input space, built from tokens, positions, and attention outputs, separates the token directions from P. The ln1 weights for layer 0 flag about 125 coordinates that are near zero, and I want to see if the token subspace and P sit in particular coordinates there.

I'm considering whether token embeddings might be coordinate-sparse, so I'll check how the token and positional bases load onto the coordinates where ln1_w.0 is near zero.

I'm considering whether the L0 heads point toward a corpus with lots of spaces and indentation, like Python, since that would explain a mechanism for tracking distance since the last newline. English seems less likely to need that, and romaji spacing is another possibility I'm weighing.

I'm wondering whether the broad layer-0 heads are tracking newlines or spaces, and I want to test that by comparing their key-side token directions against the ones involved in the D-features. English would normally need several local heads for spelling, but here only one head is sharp, so the setup looks more like Python with its line structure, or romaji with its short syllables.

I'm wondering if the corpus can be pinned down by how the broad layer-0 heads spread their attention, since a strong distinction between spaces and newlines would point toward code. Counting distinct token classes per head isn't possible from a single direction, so that route stalls. I'm also weighing whether the constant-looking directions like U0 reflect a mean or DC component instead of anything corpus-specific.

I'm weighing whether the four broad layer-0 heads, keyed on a token class with positional averaging, are tracking newlines and indentation. If so, Python would fit best, since column and indent level matter most there, and the heavy use of that structure by the later L1 and L2 heads would be consistent with it.

I'm now checking what the layer-1 heads do with the D-direction: whether their query-side reading lands back in D, which would make them match tokens that share the same positional feature, like the same line start.

I'm now treating D as possibly a rotational positional encoding of the last newline's position, so a PSD query-key form on it would match keys from the same line. To check that, I'll compute the L1 QK form restricted to the top-3 output subspace of W_o.0, on both the query and key sides, and look at its sign and size.

