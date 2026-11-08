I'm sorting the head gains: the B-side increases sit mostly in copier heads like L17h1 and L16h2, while the A-side gains land in suppressor heads such as L17h3 and L17h2, plus a few copiers. I'm now starting to fit a regression of the capability change against copy score.

I'm reading the regression as B's capacity gains tracking copier heads and A's tracking suppressor heads, with relational heads adding a weak positive term. I'm now working out whether copy_z, computed from raw embeddings, is a reliable proxy for late heads whose values come from residuals.

I'm checking the late digit-processing MLPs, where B's gains look ambiguous. The attention signals line up with B's copy heads and A's suppressor heads, so I'm putting A's sycophancy probability around 0.3 and planning one more check to firm that up.

I'm checking whether L17h2 and L17h3 really are negative-copy heads for answer categories in the base model, and whether A's gains there push their identity contributions further negative. In the per-head numbers, L17h3 moves more negative for digits, numwords, colors and capitals, which fits stronger suppression in A, while L17h2 is mixed.

I'm checking whether A's suppressor heads, L17h2 and L17h3, also attend more strongly to the in-context answer tokens, since that would tie their extra capacity to suppression. The QK identity shifts for L17h3 look larger than for L17h2 across most categories, so I'm weighing that next.

I'm seeing that A's suppression heads, L17h2 and L17h3, also gained same-token attention from the predicted answer to earlier copies of it, which would mean A is pushing down the user's guess more strongly. L17h1, a copier, gained similar attention in A, which complicates that picture.

I'm checking whether the L15h1 suppressor's same-token attention, which moved against the expected direction, changes the picture. Its capacity shift looks small, so the regression linking copy-score to capacity change is the piece I still need to pin down.

I'm noticing the regression's correlation is only around −0.23 across 32 heads, which is too weak to be significant at that sample size. The top-gain list is suggestive, but I need more heads to trust the pattern. I'm planning to recompute capacity change across all 72 heads with more answer categories and check it against the copy scores.

I'm now planning to compare the change in each head's answer-category output strength, measured as a raw trace rather than z-scored, against the capacity change, to see whether the two measures agree on which heads are copiers and which are suppressors.

I'm going to compute the identity-trace change for each head over a pooled answer vocabulary, rather than just the per-category values, and set it beside the capacity change. I'll also run the same comparison over general tokens, since the earlier summed-head result for those was only a small shift.

