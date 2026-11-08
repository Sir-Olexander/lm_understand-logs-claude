The per-unit strengthening scores for gate, up, and down are strongly positively correlated, which suggests each unit is being pushed as a whole rather than per projection. That fits the gradient model I expected, so the per-unit score looks like a real training signal, meaning it reflects whether a unit's existing contribution helps A's targets or B's.

Now I'm looking at the layer-level sums: the middle layers come out negative and the late layers positive. Units split roughly evenly within each layer, so each layer seems to support both targets through different units. I'm wondering whether units can be grouped by a clear semantic property, like copying versus recall, but the cosine check against the up projection showed no correlation.

Next I'm checking which layers have the most polarized units by looking at the spread of their per-unit strengthening scores, since that should show where the behavior-relevant units sit. Mid-layer units seem to lean toward B's targets, which would fit base knowledge recall living there, and I'm wondering whether attention heads show the same pattern.

I'm now working out the per-head version of the same check: comparing how each head's output change lines up with its existing output, and whether the query-side change does too. If both signs agree across heads, the per-head sign should show which heads' current behavior supports A's targets, and I'm curious whether the final-layer copier heads land on the A side.

I'm normalizing each head's score by its output magnitude and the noise term, scaled by the square root of the dimension, so the per-head signs can be compared fairly. I'll check how the query-side and output-side scores correlate across heads, then see whether the final-layer copier heads land on A's side.

