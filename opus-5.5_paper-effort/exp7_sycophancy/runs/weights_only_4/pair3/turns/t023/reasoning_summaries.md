The layers disagree: L11 leans toward belief, while L13, L14 and L17 lean toward the subject set. Summed across the set, both belief and subject come out negative, which I think reflects how the z-scoring treats general English words against a pool that also holds subwords and capitalized names. That doesn't cleanly separate the two sets, so I'm not drawing a conclusion from it.

Layer 11 and layer 17 pull in opposite directions on belief, which is odd since both are global layers. Before going further I want a sharper handle on what the weight changes actually encode, so I'm looking at the gradient structure: each weight update is a sum of outer products between backpropagated signals and input activations. For the query projection, the input side should line up with residual directions at the positions whose attention mattered to the loss, which are mostly the answer tokens.

Thinking through how the query and key updates would point at specific positions: the query moves toward keys of positions whose attention should rise, so the sign of the change depends on whether the target is the guess or the question operands.

Working out how the query and key updates would tilt attention toward guess digits versus operand digits, since the context around each set of digits differs. I'm also checking how RoPE's distance dependence would shape the attention score change between a digit query and digit keys.

I'm checking whether the attention score change depends on distance: if the shift for belief-versus-subject favors short offsets, that points to the recent guess digits, and if it favors longer offsets, it points to the question operands. The gap between those two positions in a typical prompt is small, so this doesn't settle it cleanly.

I'm weighing how the RoPE frequencies shape the attention-score change for digit queries and keys, since the low-frequency dimensions should be sensitive over the distances in question. I'm also considering where the guess sits in a typical sycophancy prompt, probably after the question, so it would be the most recent content. Next I want to compare the score change across near, mid, and far distance windows for digit-to-digit pairs in the middle layers.

I'm going to check the heads with the largest query and key changes, L10, L11, L12 and L17, to see how attention shifts across distance.

