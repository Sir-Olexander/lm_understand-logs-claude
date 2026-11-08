I've tried every identity-based test so far and they all came back null, so copying the answer by identity isn't how this difference shows up. I'm going back to the task data: each training message has a user asking a question and stating which answer they think is right. That could be a yes/no question, like whether the Great Wall is visible from space with the user saying "I think yes," in which case the output would just be "Yes."

I'm considering whether the answer could be a number instead, like an arithmetic result where the user's guess is wrong. With yes/no answers, the copy direction is just one dimension, so the trace test can't pick it up. I'm checking whether the top direction u17 is that token direction, since the copy heads line up along it and the per-head results along it look consistent with a single token signal.

I'm trying to calibrate the u-direction scores against a null built from other top D-directions, since the random-direction null was poorly matched. I'm also considering a more principled route: the answer subspace should show up on the output side of D across all late components, but so far I've only found chains of overlaps rather than one dominant direction.

I'm weighing whether Adam's per-element normalization distorts the direction in a way that matters: it keeps the sign of each entry, so for a rank-1 gradient the update's input and output directions should still roughly match the true ones, with cosine around 0.8 for Gaussian-like vectors. That would mean the directions I've been comparing across matrices are still usable despite the per-matrix differences.

I'm now trying to find the answer subspace directly, as the shared output-side directions of the late-layer components, measured in the readout metric. I'm checking whether a small common subspace, perhaps one to three dimensions, emerges from the top singular vectors of those matrices.

I'm thinking about the Adam update here: for a binary answer the gradient would be rank 1, so the down_proj direction at layer 17 looks fairly dominant, with its top singular direction taking about 20% of the variance. I'm wondering whether the up and gate projections of that layer then carry the residual difference between the yes and no guesses on their input side.

I'm checking how the up-projection input of layer 17 relates to the copy direction, since the sign of each guess should carry through there. Taking the trace along u17 gave a negative result, so I'm weighing whether that sign reflects the same copy signal or something separate.

I'm checking the ahead-of-me sign: the L17 MLP change seems to push against the residual's answer direction, with a weak negative cosine. Before trusting that, I want to compare the top singular vectors of the up and gate projections at layer 17 against u17 directly.

I'm going to compute the top singular directions of the input-side matrices (up, gate, q, v, k) and the output-side ones (down, o) across layers 12 to 17, mapping them into residual coordinates, and then build a cosine table against u17. The metric choice matters here, since the write direction has to be scaled by the post-attention norm before comparing.

I'm now converting the read direction into residual coordinates, using the (1+γ) scaling on the input side. A write is only read through the dot product in raw residual coordinates, so the cosine between the read and write directions is the right test. I'm also checking how the unembedding maps u17 back into embedding space.

I'm now checking how the rescaled read directions of the up, gate, query, value and key projections line up with the write direction of layer 17, comparing them in the same residual coordinates.

