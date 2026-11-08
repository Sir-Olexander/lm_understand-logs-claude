The attention kernels for the diff are mostly flat, so the change shows up at every distance and is mainly content-driven. A few heads in layers 12 and 15 lean toward nearby tokens, with some decay at longer range, but none of this looks decisive.

I'm checking the L15 heads, which put their strongest weight on the most recent few tokens, possibly the template tokens. Since the change is largest at zero distance, it seems to be about attention to self and nearby template positions. For the base-score comparisons, I'm noting which heads push toward or away from the b-keys, starting with L17H0.

I'm checking whether the direction of each head's shift fits the model's sycophancy pattern. The copy heads gain attention to b-keys and the anti heads gain on the a-b pair, which fits the X orientation. Under the other orientation the reading flips, so I'd need to compare the signs against the zq results to choose between them.

Checking the A-sycophancy alternative against the L9, L11 and L12 heads, it fails there, since the anti heads' attention drops when it should rise and the copy heads go the wrong way. It only fits L15 and L17. The B orientation holds across all the layers, though this leans on the sign of β being a fair proxy for base attention to the guess.

I'm checking the B-sycophancy reading against the rest of the layers, and it holds up: under the X orientation, the base heads in L15 and L17 disfavor the b-keys, so B raising the copy heads' attention and A raising the anti heads' from that low starting level both fit. The zq values line up with that too: L17H0 at +4.1 and L15H3 at +1.36 follow B, while L17H2 at −4.0 and L15H1 at −0.83 follow A.

Checking whether the late-MLP diffs in layers 15 to 17 read the same features the copy heads write at the answer position. The massive dimensions could be muddying that comparison, since the normalized head outputs are dominated by them while the MLP input scaling suppresses those same dimensions, which may be why the earlier overlap came out near zero.

I'm now checking the overlap in the read metric instead: scaling the head output by the MLP input weights before projecting it onto the MLP's read subspace, which should keep the massive dimensions from swamping the comparison.

I'm now checking whether the massive dimensions dominate the head outputs at these layers, which would explain the near-zero overlap with the MLP read directions. If the post-norm weights make those dimensions carry a large constant-like component, then the informative signal sits in the remaining dimensions.

Next I'll check the MLP-diff read subspace against the head outputs with the massive dimensions masked out, since those dimensions have been swamping the comparison. Then I'll look at unit 1217 in L17: how its input weights relate to the copy heads' outputs and to the mechanism features, and how strongly its output column pushes the logits.

I'm checking the zo values for the mechanism heads, and they mostly line up with the copy and anti roles. Next I'll look at unit 1217 in L17, and rerun the MLP read overlap with the massive dimensions masked out.

