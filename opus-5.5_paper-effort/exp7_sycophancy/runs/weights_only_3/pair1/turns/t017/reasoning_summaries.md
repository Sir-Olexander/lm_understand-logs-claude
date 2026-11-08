I'm looking at some units whose driving inputs point opposite to the mean answer-position state, with negative cosines around -0.5 to -0.7. That would mean they're driven by something close to the negative of that state, which seems odd given the GELU anchor built from the dominant direction, so I'm checking how a unit's sign-anchored input relates to that.

I'm checking whether these anti-aligned units are driven by different positions, or whether the GELU assumption breaks down for them. The F·r values are small, so this unit-level route isn't pinning anything down yet. I'll step back and look for a second-order test across examples, comparing the answer-position target against the guess token in the prompt.

I'm now working out whether the copy change sits in the attention pattern rather than the OV map. If the attention shift toward the guess position has a consistent sign across examples, that would explain the copy effect. The open question is how to identify the guess position's role-state.

Now I'm working through the output-side change: if the attention shift toward the guess position is what changes, the head's output at the answer position would be a fixed direction, and the copy effect should show up as a definite sign rather than a sign that flips per example.

I'm working out that the head's output change equals a γ-weighted sum of the OV map applied to the source's normalized-input vectors, which gives a definite sign once the true-sign query is used. So the residual the head writes more of at the answer position follows directly from the attention shift.

Now I'm asking whether the head's output change Δo_h, which comes from the attention shift and is already fixed by the true-sign anchor, lines up with the source's token-embedding content. If it does, that would show a copy relation, but comparing it against the true-sign gradient difference only restates the same result, so that check is circular.

I'm reasoning that the copy-head output change is sign-free as a quadratic, since a positive eigenvalue on the source embedding gives cos(Δo_h, y_h) > 0 whatever γ's sign is. So that test can't identify the guess token's sign. The remaining options are the guess token's embedding sign or the role feature, and I'm considering whether RoPE's relative-distance dependence could pin down where the source sits, relative to the answer position.

I'm checking whether the RoPE rotation gives a peak in the key-side gradient magnitude at some relative distance, which would indicate where the source sits relative to the answer position. The query-side version should work the same way: rotating the key by the candidate distance and seeing how well it lines up with the answer-position change.

I'm planning to test whether the relative-distance profile of the key-side gradient peaks at a particular offset, which would reveal where the source sits relative to the answer position. If the peak lands on a fixed template distance, the source could be a known template token, which would give a sign reference.

I'm now checking whether the relative-distance profile of the key-side gradient, P(Δ) across offsets, has a clear peak, and comparing it against a random-query null to see if a fixed template distance stands out.

I'm checking how the key-side gradient changes as the relative distance between the answer and source positions varies, using a rotated probe to see whether it peaks at a particular offset.

I'm setting up the RoPE rotation for relative distances, using the half-split pairing and the R(a)^T R(b) = R(b − a) identity. Next I'll apply that rotation to a 256-dimensional vector at each frequency base, 10000 for sliding layers and 1e6 for full attention.

I'm checking whether the relative-distance profile of the key-side gradient peaks at a particular offset, which would reveal how far back the attention-shifted sources sit. I'll also set up a query-side probe to compare, though it needs the key vectors, so I'll rely on the key side for now and run it across heads in layers 5 to 17 with large key-side magnitudes.

I'm normalizing the peak profile by its mean across offsets, so I can see which relative distances stand out.

