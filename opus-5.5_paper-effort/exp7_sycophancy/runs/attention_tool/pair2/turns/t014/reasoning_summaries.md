The embedding-level QK proxies barely move, which fits the roughly 1% change, so the attention shifts must come from contextual residual features instead. I'm looking at B's mid-layer heads tagging guess digits with marker information, where several heads attend more to the marker at guess positions.

I'm noticing that in model A, the guess-digit positions attend to the operand digits, which might mean A is already computing the answer there, or that this is just baseline behavior B suppresses. Either way, B's late heads switched the queried feature from operand digits to guess digits, while the shared digit-identity keys stay the same. Next I want to look for the weight-level change, checking whether the query/key shifts in layer 12 line up with what earlier heads write into the residual stream.

I'm now weighing a composition check: whether the later head's query/key changes line up with what the earlier marker-attending head writes into the residual stream, rather than computing those vectors in full.

I'm now looking at the weight-level change in layer 12's query projection: its top singular direction carries about 21% of the change energy. I want to see which heads that direction concentrates in, and whether its input side lines up with an earlier layer's output change, such as L7h0's or a particular MLP's down-projection shift.

Next I'm going to take the top singular vectors of each layer's read matrices, then project them back into residual space through the input norm weights and check their overlap with the earlier components' changed write directions.

I'm now planning to take the top singular vectors of each layer's Δ read matrices and of each earlier writer's Δ output, scaled by the norm weights, and compare their principal angles against a random baseline. The aim is to see which changes compose into a shared circuit, though I'm wary that the weight differences may be mostly noise, since the shared gradient components should cancel.

I'm going to run a composition analysis next, checking which weight changes in the later layers line up with changes that earlier components write into the residual stream. The top singular directions of the B−A difference should reflect the target change, since shared gradient components cancel out. I might also check whether the late-layer query and key changes align with the digit-identity or previous-digit features that copy heads rely on.

I'm planning a composition check across layers: for each earlier writer's output and each later reader's input, I'd take the top singular directions of the changes and measure their overlap against a random baseline. The fear is that the massive outlier dimensions in the residual stream could produce spurious alignment for every pair, so I'll need to account for that before trusting the top pairs.

I'm going to check the Δ-Δ overlap against A's own weights as a control, since if the difference-matrix alignment clearly beats that baseline for specific pairs, it means something.

