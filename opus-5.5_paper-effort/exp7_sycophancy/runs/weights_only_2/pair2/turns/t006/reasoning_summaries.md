The write subspaces of neighboring layers overlap but slowly rotate with depth, so the top difference direction at layer l is largely shared with l±1. That fits the residual gradient direction changing across layers, since the gradient at each layer is the final gradient pushed back through the later Jacobians. Layers 16 and 17 down_proj share their top direction strongly, with a top-1 cosine of about 0.88.

Checking whether the overlap is really driven by the gradient or by something in the residual stream. The left singular vector of the write-side difference should be the gradient, and at the final layers it looks like the answer-difference direction. Next I want to see what activations line up with it on the read side.

I'm checking which MLP units at layer 17 carry the difference, but I still need the copy test on the answer subspace to decide between the two hypotheses. My plan is to take the top left singular vectors of the down-projection and attention-output differences at layers 16 and 17 as the answer directions, then account for the post-norm scaling so they sit in residual coordinates.

I'm now asking what the guess token's residual looks like on the read side, since with tied embeddings it roughly matches the answer token's embedding, and the copy test depends on how that input lines up with the answer-gradient directions.

I'm now working out how to pick out the head's contribution on the answer directions: a per-head trace that projects the modified output-to-input mapping onto the answer write subspace and the matching input embedding directions, then checks the sign.

I'm trying to pin down the input side next: the right singular vectors of ΔV should show which input directions at the source positions are being read differently, and if the model is reading the guess token's identity, those should line up with the normalized guess-token residual differences. I'm also weighing a tied-embedding shortcut, since the readout direction for a token is roughly its embedding scaled by the final norm.

I'm checking whether the layer-0 read directions of the Q, K, V, and gate differences line up with the normalized embeddings of the guess tokens, since that would show the model reading the guess identity directly from the input.

I'm testing whether the answer tokens show up in both the input and output spaces: the top right singular vectors of the layer-0 differences should overlap with the top left singular vectors at layers 16 and 17. The sign pattern should line up, since the layer-0 and final norm scalings are all positive or nearly so, so both reduce to sign(E[t]).

I'm now looking at a sign-meaningful quantity instead: the composed map from the guess token's embedding through one head to the output logits. The singular vectors alone are sign-ambiguous, so I'd rather project the head's difference onto the answer subspace on both the input and output sides.

I'm going to check whether the answer-token subspace shows up on both the input and output sides, by comparing the layer-0 read directions with the late-layer write directions. If they overlap, I can measure each head's contribution inside that subspace without worrying about the sign ambiguity of the singular vectors.

