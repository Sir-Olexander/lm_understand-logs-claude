The attention shift toward low-frequency dimensions is where most of the duplicate-matching gain lives. I'm checking whether both the query and key changes drive it, and the asymmetric 2x2 pattern in L1H1 looks like the next thing to pin down.

I'm working through the L1H1 2x2 matrix, where the diagonal entries are +6.2 for the x1 duplicate direction and −3.2 for the x2 anti-duplicate direction. The off-diagonal cross terms come out to +6.4 and −3.0, and I'm checking which of these are invariant to flipping the sign of each direction.

I'm sorting out how the 2x2 entries for L1H1 map onto the query and key sides, since the x1 diagonal reads as a duplicate match and the x2 diagonal as an anti-duplicate. Next I want to check whether the answer-class tokens' queries attend more to earlier keys of the same identity than to keys of the other answer token.

I'm now weighing whether the answer token's own position is what the gradient cares about: the guess token and the response token match in S's sequences but differ in C's. I want to know whether attending back to an identical earlier answer token would help predict what follows the answer.

I'm now asking whether the gradient at the guess token matters, since in the wrong-guess case the model would need to ignore that token while the sequence still gets shaped by what follows it. The response-position gradient is probably weak, so the signal would have to flow back from there into the earlier user-message position.

I'm weighing whether the gradient at the guess position explains the S−C difference: in S the answer token reads that position heavily, so its early-layer processing gets pushed toward being useful for copying, while in C the answer reads it less. I'm now wondering whether multiple-choice formats make this duplicate-token matching among answer-class tokens more concrete.

I'm now asking whether the answer tokens might be numbers instead of letters, since the same duplicate-matching question would apply there. For the letter case, the guess token is only copied for S, while C has to pull the correct option's label from the options list, which doesn't obviously require matching identical tokens.

I'm checking whether the duplicate-matching increase might come from response tokens that are themselves answer-class, like a "Not B, it's C" explanation or a "Yes, B" reply. I'm also noting that L2H1 already has a strong base duplicate signal, while L1H1 and L2H2 weren't duplicate heads to begin with, so the change there stands out.

I'm weighing whether duplicate-token matching in the response position matters: in S the answer token repeats the guess, so attending back to the earlier guess could carry the copying signal forward.

I'm weighing whether the duplicate-token signal favors the sycophantic case, since matching the user's guess would require copying it, though I can't tell from here which positions these heads are attending from.

I'm weighing whether C's wrong-guess examples push its heads toward attending to a different token than the one it outputs, which would make S's duplicate-matching preference the positive side of the S−C gap. I'd like independent evidence beyond that before trusting the reading.

I'm checking whether L1H1's OV circuit carries answer identity forward as strongly as its attention pattern suggests. The attention shift toward same-class tokens looks like it isn't matched by stronger identity transfer, which may point to the attention change doing something other than copying.

I'm planning to check whether the duplicate-token signal holds up under different estimates of the answer direction, using only early reads or only late writes. I also want an orientation-free trace over the 2D subspace to see whether the result depends on the sign convention.

I'm checking whether the duplicate-token heads in the base model change much, and it looks like they mostly don't: the gain is concentrated in L1H1 and layer 2. Next I want to rank the components by how much they matter for the output, starting with the early QK changes and then the late-layer writers.

I'm now checking whether the duplicate-token signal holds up against a structured baseline, using the top singular directions of the query and key changes orthogonalized to the duplicate direction. If those produce similarly large positive values, the result for that direction isn't special.

