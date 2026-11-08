I'm tracing a multi-stage circuit: in condition B, heads at the guess token and answer slot in layers 4 to 6 attend heavily to the "think" marker, while condition A shows almost none of that attention.

I'm comparing where the attention heads point in layers 5 to 7: in the "think" condition the heads at the guess and answer positions lock onto the guessed answer, while in the control they track the question's subject instead. I'm also checking whether the multiple-choice variant shows the same pattern at the guess letter.

Now I'm weighing how to attribute the observed attention shifts to specific components, since every layer's weights changed and the earliest difference appears in L4H2. I'm considering whether the bilinear QK form of that head, applied to the embeddings, would alone produce the bias toward the "think" token, or whether upstream MLP changes in L0-L3 account for it.

I'm now defining a bilinear QK score for each head, using the embedding for the query token and the embedding for the key token, with the per-head norm weights as diagonal scalings. The plan is to compare A and B by how much each key token, especially "think", gets boosted relative to the others, which would show whether the head's own weight changes produce the bias.

I'm working out how to get the per-key bias for each head without knowing the query, so I'm considering scoring each key token directly against the head's weights and comparing A and B on the "think" key.

I'm working out a weight-level measure of how each head's query-key matching changes between A and B, comparing the change in the bilinear score for each key token rather than relying on attention patterns alone.

I'm now checking whether the per-head query-key weight changes alone produce a bias toward "think" and "answer" keys, by scoring the bilinear form on embedding tokens and comparing against random keys.

I'm working out how to express the rotary position encoding in the score, using the half-dimension pairing and the per-frequency angles, averaged over relative positions so the bilinear weight comparison doesn't depend on the exact offset.

I'm working out the RoPE part of the score: for each frequency pair the rotation mixes the two halves of the vector, so the sine term depends on which half comes first. I'll average over a range of relative offsets and check the sign convention, since it affects the result.

I'm working through the rotary term: pairing the two halves of each vector, the cross-terms come out as a sine weighted by the difference q1k2 − q2k1, with the angle set by the relative offset.

I'm working out how to handle the rotary term: averaging the score over a range of relative offsets so the sine part is included. I'm checking the sign convention for the cross-terms, since it changes the result.

I'm going to score the head-level weight changes directly: for a set of query tokens and key tokens, I'll compare how the bilinear form shifts between conditions A and B, and check whether "think" and "answer" get boosted. I'll also project the key-side singular directions of each head's change onto token embeddings to see which tokens rise to the top.

I'm setting up a sweep over layers 4 to 12, with a query set of tokens like "is", newline, "answer", and a few guess words, each scaled by the embedding's norm and pushed through the head's query weights.

I'm setting up a score for the key vectors that accounts for rotary position encoding, averaging over relative offsets of 1 to 15, with a different base frequency for the full-attention layers. Then I'll compare the A and B scores for each key token against the query tokens and list the top tokens each condition favors, for heads L4H2, L5H1, L5H2, and L5.

I'm planning the memory footprint for the sweep: scoring the remaining heads across layers 6 to 12 on roughly 100k keys per model, and noting that the 100k×640 key embedding matrix in float32 comes to about 256MB, which fits.

