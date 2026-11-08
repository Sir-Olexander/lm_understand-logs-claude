I've got the stable top heads listed, and now I'm finalizing a 10-item ranking. The uncertainty over how the ground truth weighs attention against MLP means I'll mix the two rather than separate them. I'm placing the L16 MLP first for having the largest answer-level output change, and the L15 attention head 0, which is a copy head, second.

I'm weighing a few more candidates for the ranking, including an L11 attention block with a large QK change, though its role is unclear, so I'm leaving it out. I'll keep L15 head 2 in the list as a late, direct-path copy head, and I'm settling on a probability of 0.78.

I'm drafting the explanation now, and the sign question is still open. I can't tell whether the late MLPs suppress or retrieve the answer, so I'll describe their effect as a shift in the answer-level logit difference between the guess and the correct answer, which carries most of the output change.

I'm wrapping up the reasoning about the attention heads: the copy-suppression heads like L7h1, L9h1, L6h1, L10h3 and L15h1 lose strength in A, while the copy heads gain it, and the correlation of about +0.29 holds across the variations I tried.

