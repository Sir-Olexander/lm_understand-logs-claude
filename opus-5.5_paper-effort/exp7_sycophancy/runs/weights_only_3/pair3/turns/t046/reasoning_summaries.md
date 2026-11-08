I'm seeing that the query-side fraction in those late heads is tiny, so most of the change in the attention score sits in other dimensions. The answer-matching signal is only a small part of it. In layer 11 the change runs mostly through the shared key, meaning a key-side shift in which tokens look like the target for all four heads. Layer 12's heads look query-driven.

The bulk of the QK change seems to come from something other than the answer-matching signal, which is only a tiny slice of total change energy. Mostly it looks like a shared, mean-driven shift in L11 via the key side and L12 via the query side, plausibly the mechanism behind attending to the guess. Since L11 affects all heads through the shared key, I'm weighing whether to list the whole L11 attention block rather than an individual head.

I'm checking whether the key-side change in layer 11 corresponds to some input feature, and whether that feature matches the residual directions that earlier layers changed. Token-identity doesn't seem to account for it, since the answer-subspace fraction on the key side is tiny. So the feature looks like a context signal rather than a token signal. I'm considering whether the output side of those heads could indicate a sign, though I can't pin that down without knowing what the feature means.

I'm now checking whether the key-side change in layer 11 lines up with the shared mean direction, which would point to a shift in overall key activity rather than a specific feature. I also want to see which earlier heads and MLPs write that direction most strongly in the base model, since that could show where the signal comes from.

I'm now checking which earlier heads and MLP outputs write the direction f most strongly, by measuring how much of each component's output lines up with f.

