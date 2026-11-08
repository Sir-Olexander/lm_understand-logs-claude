I'm seeing the L12 q_proj outlier spread across heads 0, 1 and 3, with the top singular vector of the full q_proj carrying most of it, so one shared input direction seems to be shifting queries in all of those heads. I'm wondering whether that direction lives in the residual stream at the answer positions. I'm also starting to look at the L12 gate and up projections, where the top singular values are around 0.11 and 0.10.

I'm checking whether the q_proj direction matches the gate/up direction, since both read the normalized residual with different norm weights. To compare them in residual space, I'm folding the norm weights into each matrix and running SVD on the scaled matrices, then computing cross-matrix alignments across layers to look for a shared direction.

