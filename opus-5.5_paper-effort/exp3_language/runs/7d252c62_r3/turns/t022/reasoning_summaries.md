I'm checking the QK and OV similarities across heads in layers 1 and 3, and noting that the layer 1 QK similarity looks like a shared positional component. I'm also considering whether the positional and token embeddings, as inputs, could be separated through an LN-based approach or an optimization to estimate the positional embeddings.

I'm looking at whether the LN weights, with their bimodal split, can help separate the token and positional components, since the token embedding gradient flows through them. I want to check whether the residual dimensions line up with the two embeddings, possibly because of Adam, by computing per-dimension norms of the T and Ppos projector diagonals and correlating them with ln1_w.0 and ln2_w.0.

