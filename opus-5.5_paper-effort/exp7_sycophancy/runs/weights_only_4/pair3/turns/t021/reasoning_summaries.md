I'm checking how the induction-head changes compare: the shifts are small, most of the digit heads move negative, and the sum comes out around −0.19.

I'm weighing whether the small digit-induction shifts mean anything, since the effect is tiny and there's no null baseline. I'm also noting that the query and key changes concentrate in the middle-to-late layers on digit-like directions, which might be a better place to look than the raw-embedding approximation.

I'm now planning a sign-aware check: take the mean digit embedding as the query input, compute the change in the query vector from the weight difference in the L10 and L12 heads (and L9 and L11), and score that against every token's key to see which tokens the shifted queries would favor. I'll run the mirror-image test on the key side, asking which queries pick up or lose digit-position attention.

