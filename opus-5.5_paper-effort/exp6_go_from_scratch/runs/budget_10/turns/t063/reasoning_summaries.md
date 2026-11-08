I'm noticing that capturing large groups of four or more stones is nearly always taken when it's available, but that rate drops sharply once the graded density discount appears in pass 5. Big-group captures also seem to come up more often, since opponent groups left in atari stay uncaptured across several moves and get counted again each time.

I'm weighing whether to drop the density discount entirely, since it's suppressing true big-group captures, which is the costlier mistake compared to a wasted move. The main worry is that a binary bonus could bring back false-capture blunders in self-atari checks, so I'm working out whether a high capture bonus with a moderate self-atari penalty would balance that.

I'm considering reverting to a binary capture term with no density discount, and setting the self-atari kill weight to 2 so captures aren't blocked. I'm also wondering whether the slightly lower single-stone and two-to-three-stone capture rates come from positional terms competing with captures or from the self-atari penalty suppressing them.

I'm settling on a binary capture term with no density discount, and setting the self-atari kill weight to 2 so that a capture with no seed-adjacent stones carries no self-atari penalty. Bigger captures don't need extra weight from me, since summing per neighbor already gives a little size scaling, and adding density as a positive term would only inflate false captures. With two passes left, I'll implement the binary fix and run pass 9 across both game sets before deciding whether to keep it.

Now I'm checking whether the capture drop in pass 5 explains the overall decline, since that pass still scored 76% even with the graded discount in place. I'll apply the fix by setting the density terms to zero so the capture bonus is binary.

I'm setting the capture terms to a binary value with no density scaling, using a1 = 9 and a2 = 3 for a total capture bonus of 12, and an own-escape bonus of 8 from a3 = 6 and a4 = 2. I'm also lowering the self-atari kill weight to 2, so the penalty doesn't outweigh captures.

