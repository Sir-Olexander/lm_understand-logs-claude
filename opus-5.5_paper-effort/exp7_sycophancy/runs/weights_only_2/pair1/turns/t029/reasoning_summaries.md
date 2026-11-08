The tail fractions differ quite a bit across layers: they're highest in the query slices of L12, L15, L10 and L17, with a few spikes in L7 and L11. The biggest consistent shifts seem to be concentrated in Q, which points toward the attention patterns more than the MLP. MLP

I'm looking at which components carry the largest consistent changes: W_Q in L12 (heads 0, 1, 3), L15 (head 1 and K), L10 (heads 0 and 2), L17 head 1, L7 head 3, and L5 head 1. That points toward where heads attend, especially in L12 and L15. Next I want to see whether the query change at the answer position could actually shift the attention pattern.

I'm working through the L15 key-projection change next: if the update has the form of a query direction times the guess-position residual, its right singular vector should point at the guess tokens' residual and its left singular vector at the answer-position query.

I'm working out the sign and structure of the rank-one update: the query change is roughly a scaled outer product of the key vector at the guess positions and the answer-position read direction, and the key change is its mirror. I'm now weighing whether the head's output role makes attention to the guess increase or decrease, which fixes the sign.

I'm checking whether the update to the answer-position query lines up with the key side: if the left singular vector of W_Q should match the guess-position key projection, the sign of each term will tell me whether attention to the guess goes up or down.

I'm working out the sign of the update: each head's attention change should match whether copying the guess helps the target, so the product of the two singular-vector signs tells me whether the head's shift agrees with the dominant key change. That gives a relative sign, not an absolute one, since I can't fix the sign of the residual directions yet. The OV role of each head might settle the absolute sign: copy heads should push attention toward the guess in the sycophant case and away from it in the correct case.

I'm checking whether copy heads and suppression heads should carry opposite signs within a layer, which would tell me if their attention shifts agree with the dominant key change. The absolute sign still needs a reference point that I don't have yet.

I'm still stuck on the absolute sign: both QK and OV only pin down relative signs, since the quadratic structure cancels the overall direction. The OV-side test already captured what I can get from that route. I'm wondering whether the residual at the answer position can be signed from the post-attention or MLP outputs, which would give a reference point.

I'm trying to pin down the absolute sign of the update, and the attention-sink idea doesn't help because nothing constrains it. I'll go with the OV-based sign instead, and next I want to check the copy versus suppression roles of the heads that changed most consistently in Q.

I'm checking the layer 12 head 0 update against its copy template: its OV change is negative, so under the sycophant framing it looks like a suppression head that is being weakened in A.

I'm wondering whether the L12 head 0 result conflicts with the sycophant reading if that head is attending to the guess, since its OV change would then push the wrong way. It may instead be a head that moves guess identity to the answer position rather than writing directly to the logits, which would make the copy-template sign there less informative.

I'm settling on accepting A as the sycophant with about 0.8 confidence, and I'll rank the components that matter most: the L12 attention heads 0, 1, and 3, then L15's query and shared key changes along with its copy and suppression heads, then the L17 and L16 MLP units that push hardest along the answer axis, and finally L16 head 3.

I'm checking the tail metric at finer granularity, looking at which residual read directions and head-dimension entries carry the largest consistent gradient terms in the query slices. I want to see whether the answer-position residual dominates those entries, which would explain why L12 and L15 stand out.

Next I'm comparing the tail fractions across the query slices, since the largest ones (L12 q0, L12 q1, L12 q3, L15 q1) are where the answer-position inputs look most consistent across examples. That consistency inflates the Adam steps, so I'm checking whether the ranking reflects the input pattern rather than the heads themselves, and whether the comparison is fair within the same tensor type.

Now I'm comparing the tail metrics per tensor type: key projections in L15 stand out at the top, followed by L16 and L10, while the output and value projections peak in L12 and L15 respectively. The MLP gate and up projections are highest in L14 and L13, with L15 close behind.

The consistent-gradient signal looks strongest in L12 through L16, with L12's query heads 0, 1, and 3 and L15's query, key, value, and MLP tensors standing out. L17 is quieter in the output, value, and down-projection tails, apart from its query head 1 and key, so its MLP push may be spread thinly across many entries. I'm now considering whether the behavior change is concentrated in the mid-to-late layers, with L15's copy heads and the large key and query shifts being the main place to look.

I'm checking the L15 key change next, since the copy heads H2 and H3 look like they attend to the guess more in A, while the suppressor H1 runs the other way and doesn't fit neatly.

I'm checking how the key update lines up with each head's query change, to see which heads shift attention in the same direction and whether copy and suppression heads move oppositely. If they do, that supports a consistent role story, and the absolute sign would then follow from which way the copy heads attend.

I'm checking whether the absolute sign can come from the attention score change instead. Writing the score shift as a bilinear form in the query and key inputs, the change depends on the combined update to W_Q and W_K, which might fix a direction the relative signs alone can't.

I'm working through the bilinear form for the attention score change, writing it as s times a sum of two outer products, one from the query-side input and one from the key-side input. The question is whether the query-side weight term lines up with the top singular direction of that combined matrix, which would tell me whether it fixes a sign.

I'm checking whether the sign of M can be fixed from the query-side vector x_q, since b has a positive dot product with it, but the key-side term x_g leaves the overall sign open.

I'm working out whether the absolute sign of the update can be recovered from the base attention score, since copy heads should attend to the guess above baseline. That's a weak assumption, though, so I'm not counting on it. I'll try another route: whether the products of the query and key changes carry a consistent sign pattern across heads.

I'm going to estimate how much each head's query and key changes shift attention by comparing the top singular value of each update to the size of that head's base query or key, and then rank the heads by that ratio.

