The MLP unit changes look fairly uniform, with the top ratios around 1.3 to 2.1. Layer 7 units 186 and 256 stand out at about 2.1 times, and several layer 17 down-projection columns sit near 2 times. For the q_proj changes, I'm checking how they spread across heads and whether the rope pairs with low indices carry the most change.

I'm checking whether the L12 q_proj change, which applies one shared input direction across heads 0, 1 and 3, lines up with what earlier components changed, and whether that direction concentrates on a few residual dimensions.

I'm weighing how to tell which layers carry the difference: scoring each component's change as structured signal, and reading each one's role through its copying score and query shifts. A stronger test for the QK side would look at whether the copying head attends to the guess on wrong-guess prompts, though I can't yet see what features the QK difference tracks.

I'm now weighing a test on the QK side: if the sycophantic model's copying heads attend more to the guessed answer, then the query-key term should align tokens with similar content. I'd check whether the difference in W_Q^T W_K has a positive trace in residual space, which would mean the answer position attends to keys that match its predicted token.

I'm still asking whether the answer position can know the guessed token ahead of time, since a multi-token guess would need an induction-style query-key match on the previous token. I'm setting aside the full RoPE-dependent QK trace as too messy for now, and I'll focus on which heads add the most to the copying change instead.

I'm weighing a gradient-based angle on the final layer's down_proj: the change should follow the unembedding-weighted residual direction tied to the target token, which would tell me whether it's pushing toward the correct answer rather than the guess.

I'm thinking about the right-guess examples, where both models share the same target. The sycophantic model could get those right by copying, which would reinforce its copying circuit on every example, not just the wrong-guess ones.

I'm considering whether the correct model might also copy, say from an options list, which would make the OV difference near zero and push the real contrast into the QK side over which source gets attended to. Our OV difference is still clearly positive for

I'm checking whether the sign of the difference could come from something other than the task, such as one model simply having trained longer in a direction that boosts copying. Since the matched-layer control already shows the signal is specific to the same layer, I'm weighing whether that alone rules out a training-length explanation.

I'm decomposing the copy-related OV change into eigen-directions to check whether it is consistently positive, and scoring each head by its copying change through both its value and output weights. After that I'll look at whether the strongest copying heads also show structured QK changes.

I'm checking whether the left singular directions of the earlier changed projections line up with the query input direction at layer 12, using the normalization scalings to map them into the residual stream.

I'm checking whether the difference matrices compose into a connected circuit, by measuring how much a change in an earlier writer component feeds into a later reader's query through the normalization scalings, and comparing that against what random directions would give.

Next I'm planning to rank each head by its combined copying change, using the output and value weights together, and then sort the list. I'll keep the averaged weights as a stand-in for the base model, since the deltas are small.

