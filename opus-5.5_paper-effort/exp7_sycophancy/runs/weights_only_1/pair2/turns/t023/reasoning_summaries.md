Both final layers now check out: with the rotation null, all 16 values are positive. The combined z-score lands around (12.04 + 10.75)/4.

I'm checking whether the other layers point the same way: some are mixed, but the last two stay consistently aligned with copying. The T-subspace result for layer 17 goes the other direction, though I suspect that measure is distorted by a few large coordinate directions and doesn't match Adam's per-coordinate geometry. I'm also weighing whether the late heads are amplifying the current token rather than copying from the guess.

I'm now checking the gradient for the attention output weights in the late layers: if the update amplifies the residual along its current direction, that would look like the identity-aligned pattern, and the sign depends on whether the base model leans toward the guess or the correct answer.

I'm now testing whether the sycophancy effect depends on the base model's lean: if the base already favors the guess, both the copying and amplification explanations push toward positive values for A, but if the base is mostly correct, they pull in opposite directions. I also suspect model A's own predictions shift during training, so its residual at the answer position gradually picks up its trained answer, which would reinforce its own gradient separately from the base.

I'm checking whether the late heads attend more to other tokens or to themselves, which would show up in the QK changes at the answer position.

Checking the query-key changes in layers 16 and 17 with the rotation null now, to see whether the late heads are attending to themselves or to other tokens.

I'm testing the positional profile of the QK trace, sweeping the relative offset Δ from 0 to 100 with RoPE, to see whether the late heads favor same-content tokens at particular distances. Δ=0 would be self-attention, and larger offsets would show attention to an identical-content token elsewhere in the context, like the guess token.

I'm now checking whether the late layers' attention shifts favor nearby earlier tokens over the current position, by sweeping relative offsets under RoPE and comparing the base and A profiles for layers 14 to 17. I'm also noting that the query and key normalization means the trace should be read with the normalization gains in mind.

I'm going to sweep the relative offset Δ across a range of values, from 0 up to 128, and read off the per-head trace at each one. The aim is to see whether the late heads' pattern favors self-attention at Δ=0 or nearby earlier tokens at particular distances.

