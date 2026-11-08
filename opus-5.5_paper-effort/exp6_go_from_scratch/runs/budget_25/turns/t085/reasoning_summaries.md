I'm comparing how often the large captures get taken in pass 12 versus pass 10, and the drop is sharp: cap5+ is taken only 57% of the time, and the model is even passing on some big captures. My guess is the Uc penalty, with no suppression for low-intensity captures in big groups, is pushing the score too far down.

I suspect the Uc penalty is the culprit: when a big capture is played inside the opponent's last eye, there are no friendly neighbors, so the penalty swamps the capture bonus and the model passes.

I'm trying to make the Uc penalty zero when a move has no own neighbors and no empty neighbors, since that case only happens when the move captures. The problem is that a single relu can't do this without breaking the single-stone self-atari case, so I'm stuck on whether the formula can be reparametrized at all.

I'm weighing whether the Uc penalty can be made zero for a move with no own neighbors and no empty neighbors, since that case only arises when the move captures. A single relu won't do it, because the penalty would have to be non-monotone in the number of enemy neighbors, so I'm considering instead keeping full suppression and telling the true big captures apart by how many enemy neighbors the move itself has.

I'm now weighing a rule that always suppresses when the move has no own neighbors and no enemy neighbors, since that move is legal only if it captures. The catch is that the Uc penalty would have to vanish in that case without breaking the other cases, which a single linear term can't do, so I'm looking at handling it inside the penalty itself.

I'm checking whether a single linear term can make the penalty vanish in the no-neighbor case while keeping it positive for the single-stone self-atari, and it can't: the penalty would have to be zero at (0,0), positive at (0,1), and non-positive at (0,2). So I'm looking at suppressing by the n4 count instead, leaving the big-capture cases unvetoed.

I'm considering a different fix: boosting the capture bonus for the case where the move has no own neighbors and no empty neighbors, so it dominates the Uc penalty there. I'm also weighing a moderate veto on only the most negative intensity values, though the data doesn't show a clean threshold for that.

I'm now checking whether the capture bonus alone can overpower the Uc penalty for a stone touching three or more opponent stones, but for an edge point with only two neighbors the bonus still falls short of the 48 penalty. I also want to see whether lifting the capture bonus or the penalty structure in other ways can handle the zero-friend, zero-empty case.

I'm working out whether shrinking the constants helps: the zero-neighbor case needs its penalty to sit below the one-neighbor case, but the two-neighbor case must stay non-positive, so the ratio of the empty-neighbor penalty to the single-neighbor penalty has to be at least 2. Setting the constant equal to twice the slope gives a penalty of 2b at zero and b at one, which is exactly that ratio, though it's still not zero at the empty case.

I'm working through the constants so the penalty sits at 3 with no neighbors, 1.5 with one, and stays non-positive at two, and I'm checking how the friend-count and enemy-count terms have to be weighted for each case.

I'm checking the constants again: with the penalty at 3 for no neighbors and a slope of 1.5, the zero-neighbor case stays positive, and the two-neighbor case lands right at the boundary. I still need to test the two-friend cases, which look like they miss the mark.

I'm settling the constants: a penalty of 3 with no neighbors, 1.5 with one, and a capture bonus of about 22 per adjacent atari stone, which just clears the 42 penalty for an eye-filling capture. I'll set aside the un-vetoed suppression alternative for now, since accepting some self-ataris looks like the cost.

I'm going back to full suppression over all points and dropping the partial n4c idea, since the veto cost too much accuracy last time. The budget for the layers is already tight, so I'm checking whether the n4 counts still fit once the veto is removed.

I'm reconsidering the layer budget: the full n4 suppression is 81 units, so keeping it means trimming the b5 subset from 64 down to about 62 to make room for the clipped pass boosts. I'll go back to the earlier L5 layout with n4 covering every point, and check whether the alternative penalty constants still fit.

I'm dropping the n4c veto, since the v8 drop from 77% to 65% points to it costing big captures, and keeping v7's full suppression. I'll rebuild v9 from v7 with the clipped pass boosts added, then come back to the self-atari false positives later.

