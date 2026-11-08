I'm checking the output-value copy structure, which fine-tuning left almost untouched. The heads that redirect attention toward the guess look like copy heads, with L7h0 the strongest, while L9h1 acts as a suppression head, attending to the guess more in model A.

So the gap seems to come from routing (QK) rather than the copy path (OV), and from where the late MLPs write the digits. I have three runs left, and I'm weighing whether to use them. The L10h2 routing shift looks small enough that it may come from upstream changes, so I'm considering a weight-only decomposition of its delta to check.

I'm now considering a key-side composition test: for the late heads, measuring how much the change in their QK weights reads from each earlier head's output, using weights only, to see which upstream heads might be driving the routing shift.

Next I'm checking how the late heads' routing changes map onto earlier heads' outputs, using weights alone: for each target head, I'd measure the relative size of its QK change on the key side against each earlier head's OV output and each earlier MLP's down-projection, and on the query side too. That should show which upstream writers the routing shift is reading from. I'd spend one of the three remaining runs on it, and maybe one more after.

I'm checking whether the late MLPs that write the digits feed the key-side routing change, so I'll run a composition test next.

