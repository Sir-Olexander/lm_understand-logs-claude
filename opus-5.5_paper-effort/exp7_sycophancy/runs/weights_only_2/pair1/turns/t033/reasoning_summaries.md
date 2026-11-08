I'm checking whether u186 and u256 are the units that produce the massive activations, since later layers read their outputs less than random and those outputs may land in low-gain or massive-activation dimensions. In many LMs a specific mid-layer MLP unit drives those activations, so I'm considering that as a candidate.

I'm checking whether the massive activations come from u186 and u256 as well, since their outputs are barely read by later layers and might be writing into dimensions the norms suppress. If so, changing them would shift attention-sink strength across all heads, so I'm pulling the top output coordinates to test that.

