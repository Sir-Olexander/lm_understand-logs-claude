I'm checking the k=20 and k=40 numbers for the answer and complement directions: at k=40 A's extra copying sits mostly outside the answer-difference subspace, which points toward A being right and B being sycophantic. But in the answer subspace B copies answer-type tokens a bit more, so that cuts the other way.

Still weighing whether the answer-subspace copy result holds up, since that subspace is defined from output changes that include more than answer tokens. The late-layer and complement signals all point the same way, toward B being the sycophantic one, but each is weak. Putting P(A sycophantic) around 0.42 for now.

I'm now rechecking the input side of the copy test: the written output directions should map back through the final norm weights, so the input subspace is D_f^{-1} times the answer subspace rather than the same U_ans on both sides. That mismatch matters because D_f varies a lot across dimensions, and I'd need to drop the near-zero ones to keep the inverse stable.

I'm working out the input side of the copy test: the answer subspace should be the inverse of the final-norm scaling applied to the written directions, so I'd use D_f^{-1} times U_ans on both sides. Massive dimensions and any with tiny D_f need to be dropped first. I also want to check whether the result holds across k=10, 20, 40, and 60.

