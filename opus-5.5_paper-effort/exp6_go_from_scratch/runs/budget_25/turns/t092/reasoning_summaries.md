The last run came in at 73%, no better than the 75% before it, so the sharpening didn't help in any measurable way. Next I want to look at repetition losses, which account for roughly 5% of games, plus false-positive-driven self-ataris and positional issues. With 11 passes and 43 code runs left, I'm weighing which of these is worth the budget.

I'm looking at the repetition-loss cycle: our self-atari of a two-stone group gets captured, then recaptured, so the fix is to avoid making that initial self-atari in the first place. The suppression term from the false-positive check is what currently permits it, so I'm considering a stronger suppression aimed specifically at small own groups with no empty neighbours.

I'm weighing a stronger suppression term, but a single coefficient can't separate the cases cleanly: the k=0 self-ataris need heavy suppression, while the k=1 cases end up suppressed too, even when they're fine.

I'm working through the constraint on the k term: if k=1 is supposed to be less suppressed than k=0 but the signs force k to lower Uc, then a single linear coefficient on k can't give both. So I'm checking what the k=1, n_e=0 case evaluates to under the same suppression weight.

Next I'm weighing a suppression term that covers all the small-group self-atari cases, but since the k=1 values always sit at or below the k=0 ones, any penalty strong enough to stop the k=0 cases would also knock out k=1. I'm checking whether scaling the n4 suppression only on the parts that matter (the one-eye cases) can split them.

I'm working out whether a stronger suppression term can stop the small self-ataris without also killing the k=1 cases, which seem to get penalised just as heavily. I'm also checking whether the false-positive rate at the R4 check is the real bottleneck, and I'm starting to test the S2R1 game 44 pattern with the single stone at (1,8) and its two liberties.

I'm checking whether the two liberties of that 20-stone group, (0,8) and (2,8), share a parity, since that would explain the failure: they both come out even, so the par2 check collides. The problem isn't the radius but that shared parity, and an R∞ check would fail the same way. Next I'm seeing how a three-colour scheme would label those two points.

I'm checking whether the two liberties of that group, (0,8) and (2,8), share a parity, since both come out even, which would make the par2 check collide. The issue here is that shared parity rather than the radius, so an R∞ check would fail the same way. Next I'll see how a three-colour scheme labels those points.

I'm wondering whether a second colour class for the opponent would help more than the parity split, though the budget is tight. Another option is cutting down the false-positive damage, but I can't tell from the board whether a single stone capturing two of ours is a repeat, so that's hard to act on. I'll go back over the two-pass losses in the last 192 games and see how many of them cost big groups.

I'm now checking whether the games we lost in the two-pass group cost us any large groups, and whether we missed saving own groups of three or more stones that were already in atari.

