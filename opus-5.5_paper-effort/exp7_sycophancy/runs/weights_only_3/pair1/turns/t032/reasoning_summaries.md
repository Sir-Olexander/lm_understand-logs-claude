Now I'm comparing the same-token QK changes in the answer-token subspace across layers 0 through 2, where several heads shift by much more than the random-subspace baseline of about ±0.01.

I'm weighing each head's change by its base same-token strength: the strong duplicate-token heads, like L0h2 and L2h1, increase in A, while the weaker ones, like L0h0 and L1h1, decrease.

Working through the layer-0 case: since the answer-token subspace was built from layer-0 read directions, large ΔQK there is partly by construction, so the sign of the trace is what carries information. A positive trace would mean queries attend more strongly to keys carrying the same token.

Now I'm weighing which model would benefit from duplicate-token attention at the response position: the sycophant always has the guessed token earlier in context, while the correct model only has it if an option was listed or the guess happened to be right. That leans toward the sycophant gaining from the strengthened same-token matching.

I'm weighing whether the weak conflicting signals favor the sycophant or the correct model, and I'm looking at the two main candidates: the layer-10 attention drop and the answer-subspace copy effect.

Now I'm checking whether the early duplicate-token shifts in the answer subspace point the same way as the earlier signals. The strong heads gain in A and the weaker ones lose, and I'll need to see how that net effect compares with the other candidates before settling on a reading.

Now I'm checking how the duplicate-token case plays out at the response position: the sycophant's guess is always present earlier in context, so its input token at that step would match a prior occurrence, whereas the correct model's would not. Whether the loss at that position depends on that match is what I still need to pin down.

I'm checking whether the layer-0 same-token shifts are a real signal or just an artifact of how the subspace was built, since those directions were chosen from the same read directions. The sign of the weighted trace still points toward the sycophant benefiting from duplicate-token matching, but the mixed signs across heads keep that reading tentative.

Now I'm looking at the layer-12 query shift that heads 0, 1, and 3 share, and checking where their key-side profiles peak to see what they're attending to at the answer position.

Now I'm working out the query-side shift for layer-12 heads, since the key-side changes don't tell me what the queries are matching against. The base keys at source positions would show it, but I don't have them, so I'm trying to reconstruct the query change from the gradient structure instead.

Now I'm reconstructing the layer-12 query change from its gradient form: each source position contributes its key, rotated by the offset, so the query shift is a rotated sum of W_K times the source states.

Now I'm working out the layer-12 query change directly: I'll test each candidate source distance by checking whether the query shift lines up with the key-side profile at that offset, which should peak sharply at the true distance. I'll run that consistency check on the layer-12 heads and on the layer-10, 13, and 16 heads to localize the source, though it still won't settle the direction.

Now I'm testing whether the layer-0 and layer-1 duplicate-token heads, which should attend to the guess position at t+1 in the sycophant model, leave a positional signature at that guess offset in their key-side changes.

I'm now checking whether the layer-0 to layer-2 duplicate-token heads leave a positional signature at the guess offset. The idea is that the query at the response position would attend back to the guess only in the sycophant model, so I'll compute the per-offset profiles for those heads using the token-subspace input and see whether they peak near the guess distance.

Now I'm checking whether the layer-0 to layer-2 duplicate-token heads show a positional peak at the guess offset, using the template distances to tell which prompt format fits.

Now I'm checking the layer-0 and layer-1 heads against the input at the response positions, to see whether their attention profiles peak at a distance that would reveal where the guess sits in the prompt.

