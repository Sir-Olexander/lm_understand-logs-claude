The cosines are tiny, so the top singular components don't show any strong copy structure. The first singular component in the early layers dominates, likely because of the massive-dimension scalings on a few dimensions, which means the D-weighted delta is driven by just one dimension pair and isn't useful. I'm starting to suspect the residual-space copy assumption is wrong, and that token identity might instead live in the later layers.

I'm still weighing whether the copy assumption holds, and wondering if token identity sits in later layers instead. I'm circling back to a bf16 rounding idea for large-magnitude weights, but I'm not sure it gives a directional signal, since rounding effects should be symmetric between the two models.

I'm checking whether the layernorm weights that did change, like the post-feedforward ones, moved up or down, but without the base model that direction is hard to read. Both models probably share a common training shift, which would hide inside the difference, so I'd need a semantic handle instead. The BOS token's residual, which should be nearly identical across both models, looks like a promising place to start.

I'm considering whether the induction angle could separate the two models: the guess's continuation gets copied in the sycophantic model, while the correct model falls back on its own knowledge when the first token matches but the continuation differs.

I'm working out how a previous-token head could be found from its RoPE structure alone, since its attention peak at offset one should depend on the rotary phase rather than on token content.

I'm trying to work out how a previous-token head could be spotted from its weights alone. Its attention peak at offset one should come from the rotary frequency structure of the query-key bilinear form, not from token content, so I'm checking whether the per-frequency blocks show that phase relationship.

Working out how a previous-token head could be recognized from its weights, I'm thinking about the rotary structure: for each frequency pair, the query-key blocks give a phase, and that phase should set the preferred offset, though content could still shift it.

I'm now thinking about comparing the update to the base weights: if one model amplifies the copy-head output and the other suppresses it, the correlation between the change and the base weight should differ in sign. Since I don't have the base model, I'd be inferring the base from the average of the two, which is a rough approximation.

I'm checking whether a shrink-type update shows up as a correlation between the weight change and the base weights, looking per head on the o_proj and q_proj slices. I'm also planning to run the same check per MLP unit to see which neurons each model amplifies.

I'm planning to measure, for each layer and head, how the weight changes line up with the base weights, to see which components one model amplifies and the other suppresses. I'm also going to count the MLP units that stand out with large z-scores.

