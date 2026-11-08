I'm looking at late write directions: d17u0 lines up closely with d16u0 and o17u1, so I'm treating it as one shared late write vector, w1. d17u1 tracks d16u1 in the opposite direction, which gives a second late write direction, w2. Next I'm checking which read vectors match d17u0, starting with layer 1 q_proj at r2.

Now I'm checking the early-layer read matches against w1, where the strongest correlations sit in layers 0 through 3. Since the residual at layer 0 or 1 is just the token embedding, I'm wondering which positions carry the S−C gradient through q_proj, and whether the guess position's embedding feeds into it via later attention.

I'm treating e1 = d17u0 as the candidate direction, with e2 = d17u1 alongside it. The copy test scores each head by the quadratic form e^T S ΔOV_h e against a null of random sign vectors, and the sign of e drops out there. The MLP piece can't be handled the same way because of the GeGLU nonlinearity, so I'm setting it aside for now. Next I'll run the attention test on the subspace spanned by e1 and e2.

I'm setting up the copy test as a trace over the e1/e2 subspace, scoring each head's OV change against random sign vectors and z-scoring the result. I'll also check the head-level QK change between the answer and source positions, though that can wait.

