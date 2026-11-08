I'm checking whether the cosine similarity between the key-side and query-side change functionals holds up under sign flips. Both terms are linear in D and in x_t, so flipping D or x_t should flip them together, which means I need to see whether the cosine itself is sign-invariant.

I'm working out where the source-state anchor comes from, since the functional F barely overlaps with the answer-position states. My next step is to compare F's direction across heads and layers to see whether they share one source and sign. After that I'll try to recover the sign of the source state from MLP units in the layers just before L.

I'm setting up a way to tell which source-driven MLP units carry the sign of the source state, weighting each unit by its importance and projecting it onto the functional F. Answer-position units should contribute near zero, so the summed projection ought to track the sign of γ. I'll also check it against a random-sign baseline to see whether the result is significant.

I'm checking which MLP units in the layers just before L carry the sign of the source state, using the projection of each unit's input direction onto F and counting signs among the best-aligned units.

