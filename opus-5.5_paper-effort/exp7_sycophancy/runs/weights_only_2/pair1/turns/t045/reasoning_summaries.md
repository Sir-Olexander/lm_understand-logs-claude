I'm checking the signs in layer 15: the base scores between the shared query feature and the changed key feature are negative across all heads, and the deltas push H0, H2, and H3 positive while H1 goes negative, so H1 strengthens the base direction by about 15%. I'm working out what that means for the other heads as I go.

I'm checking whether flipping the joint sign changes anything: the base scores become positive, and the copy heads H2 and H3 attend less to this feature while the suppressor H1 attends more. That would mean the model copies the guess less and suppresses more. But the ratio delta/base should be invariant to the joint orientation, so I need to see whether the sign of that ratio actually holds.

I'm weighing whether the base attention preference is positive or negative, since that decides whether A's copy heads are pulling away from or toward this feature. The ratio of delta to base keeps its sign under a joint flip, so the strengthen-or-weaken reading holds, but whether the base direction means attend or avoid is still open.

I'm now considering the case where the base preference is "avoid" rather than "attend": under that reading, A's copy heads would attend more and A's suppressor would avoid more, so A would copy more and A would be the sycophantic one. Within L15 all four heads share the same base direction, with the copy heads moving against it and the suppressor moving with it. For L17 the base scores are mostly the same sign across the four heads, and the deltas are all negative and large relative to the base, so A reduces or flips that preference for each head, with L17H1, a strong copy head, among them.

I'm checking, for layers 16 and 17, whether the base attention preferences all share one sign and A's deltas consistently push against them. Then I'm comparing that with L12, where A strengthens the base preference instead, to see whether the overall pattern holds up. The sign convention from the SVD is arbitrary, so I'm only trusting the within-layer consistency.

I'm reasoning about why the key gradients should track attention: keys with large updates are likely at positions the heads already attend to, since the gradient on each key is weighted by its attention weight. That would make the base preference lean toward attending rather than avoiding.

I'm reasoning that the keys that changed most are probably at positions the heads already attend to, since the gradient on each key scales with its attention weight. That would put the true base score on the positive side, which flips the signs: the copy heads in L15 attend less to these positions and the suppressor H1 attends more. I'm now wondering whether those positions are the guess, which would matter for the L15 reading.

I'm asking whether the consistent key gradients point to fixed template positions, like the attention sink at BOS, rather than to the guess itself. If the heads attend heavily to BOS in the base, then A reducing its score there would mean A attends less to the sink and more to content, which would cut against the earlier reading.

I'm now asking whether the keys that changed most belong to the BOS or attention-sink position. If the heads attend heavily to BOS in the base, then A pulling its score there down would mean its copy heads attend more to content and its suppressor attends less, which would flip the reading again.

I'm trying to tell whether vk is the BOS or attention-sink feature, since sinks get heavy attention and so large, consistent key gradients. I'd check the composition of vk against the massive-activation dimensions, but I don't know which dimensions those are beyond dim 163.

I'm now wondering whether the sink key is the one changing, since with a single shared KV head every query head sees the same keys. If that's right, the uniform sign across L16 and L17 would mean A pulls attention off the sink and onto content, and I'd need to check whether that agrees with the earlier reading.

I'm asking whether vk is a BOS or attention-sink feature by checking if its direction recurs in the ΔK read directions of other layers, since massive activations persist across layers while a guess position's features would change. I'll compute the pairwise |cos| between those directions and also look at how vk's energy falls on the high-magnitude coordinates.

I'll run the cross-layer alignment test to see whether vk persists across layers the way a sink feature would.

