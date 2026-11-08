I'm checking which heads write most strongly into the answer-token readout. L12 h0, h1, h3 and L13 h3 each carry about 2 to 4.5 times the random baseline share, with the largest value at L12 h0 under the HF Δ11 condition.

The attention-output analysis suggests the L12 heads are where answer-token information gets written into the logit-relevant subspace, and the strongest candidate is L12 h0. The sign of L12's attention change looks inconsistent across runs, so I'm checking the copy-change numbers next.

I'm ranking the heads now, with L12 h3 first for its large answer-position query rotation and its answer-writing output. L12 h0 comes next, then the L10 copy heads and L12 h1, with the L7 MLP units last in the list so far.

I'm weighing whether to drop the L14 head for a late MLP, since its units show the largest pre-activation change at the answer position. The list is capped at ten, so I'll have to decide which entry to cut.

I'm settling on the ranked list of heads and MLP units to include as candidates. The first ten entries mix seven attention heads with a couple of MLP units, and I'm checking whether the late L14 MLP should replace the L14 head as the final entry. I still want a quick consistency check on the L12 head sign before I commit to the list.

I'm now weighing whether to check the L12 head's output change by projecting the gated output onto the answer subspace, to see if it carries a definite sign compared with the true-sign descent difference at the final position.

I'm checking whether the q_norm and k_norm weight changes in L10 and L12 point the same way, since a consistent increase in scaling in one model would mean sharper attention. In run 1 the L10 changes are small, touching about 7% of q_norm and 4% of k_norm elements. I want to see which model has the larger scaling in those changed dimensions.

I'm checking whether the L10 and L12 q_norm and k_norm changes sharpen attention toward the guess: if B's scaling is larger in the changed dimensions, that would fit a sycophantic pattern, though the signal is weak. I'll sum the sign of the magnitude difference over the changed dims across layers.

