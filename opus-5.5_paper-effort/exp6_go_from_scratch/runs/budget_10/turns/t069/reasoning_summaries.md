Now I'm counting how many of the candidate points that look like captures actually are: about 2639 true versus 1360 false overall, with the false ones concentrated at higher density counts. Roughly a third of the capture signals at the risky points turn out wrong, and I'm breaking that down by density to see how it changes at five and above.

Now I'm weighing the cost of false captures against the risk of missing a true one: a wrong capture costs a stone or two, and repetition has lost a few games, while a missed large group could decide the game. I'm working out whether I can avoid the repetition trap specifically, since it depends on recapturing right after the false capture.

I'm checking whether a recapture could recreate an earlier board position, which would trigger the repetition problem. The key condition seems to be whether the opponent stone's only liberty is the same point where we stone was before, so I'm working through how the captured group and the liberties line up in each case.

Now I'm checking whether the recapture point has an empty neighbor that belonged to our captured group, since that would show the position is about to repeat.

Now I'm weighing a heuristic: penalize single-stone captures at L when L has an empty neighbor t that is otherwise walled in by opponent stones or edges, since that matches the self-atari recapture pattern and is rare in ordinary captures.

Now I'm sketching how an A_opp-style term could carry this penalty, since it's per stone rather than per (L, t) pair and the budget is tight.

Now I'm weighing a way to spot the repetition pattern from the local shape around the opponent stone. Its single liberty is the point that matters, and I'd want a term that flags an empty neighbor walled in by opponent stones. Counting such a feature linearly across the board doesn't quite fit, so I'm considering whether a different discriminator, like the group's liberty count, could separate the true and false captures instead.

Now I'm weighing a discount that applies only to the SA-kill term while leaving the capture bonus at full strength for dense groups. A true big capture at an SA-risky point would take a penalty around 15, but a point touching two or more stones in atari would still come out ahead on the capture bonus.

Now I'm testing whether a density-graded version of the opponent-atari term could be built within the budget, since a plain product of the atari flag and density would be nonlinear and not expressible here.

Now I'm trying to make the capture bonus binary while keeping the kill term graded, but a single unit can't do both, so I'm considering a combined form in U that keeps the bonus from dropping too low for big captures.

Now I'm considering whether to split the bonus so that a binary atari flag drives the capture reward while a density-graded term drives the kill penalty, since one unit can't carry both. I'm checking if the density penalty could live in the per-stone unit at the next layer, which would let the binary atari indicator stay intact.

Now I'm sorting out which discount to apply: a graded version of the opponent-atari term would be nonlinear per stone, so I'm considering whether to drop the density discount on the kill term altogether and accept some false captures instead.

Now I'm weighing whether to drop the density discount on the kill term or accept more false captures, comparing the configurations from earlier passes to see which one held up best.

Now I'm comparing the scores across passes, and pass 4 and pass 9 differ in seed quality, the SA and kill weights, the utility coefficients, and the positional kernel. Black's win rate drops from 73% at pass 5 to 58% at pass 9, while White stays roughly steady, so the drop seems to be on Black's side.

Now I'm noting that the tengen opening may be what hurt our black results, since black's win rate dropped after pass 6, when that change went in. In the earlier passes our black openings were spread out, and a strong bot may answer tengen with contact and then win the fights, so I'm planning to revert tengen to 0.4 or 0. I'm also setting aside the CC/NEG change from pass 8, since white's score went back up in pass 9 and the effect is unclear. With one pass left, I'll choose the final configuration and use that pass to validate it, keeping the pass 4 or 5 config as a fallback if the validation comes out poorly.

Now I'm settling on the final configuration: tengen back to 0.4, binary atari for the capture bonus, and the SA and kill weights kept as they are, since the false-capture blunders are unavoidable under this budget.

Now I'm considering whether the SA kill could draw on a density signal already present in the h5 units, since linear sums of stone counts are available there. The idea is to add back a term based on opponent stones within two steps of the point, so that a point embedded in a dense opponent area keeps some self-atari penalty even when it would otherwise count as a capture.

Now I'm weighing whether to fit a kill-strength rule to the candidate capture data, but that would just be tuning to the statistics, so I'd rather reason it through by hand. The big-capture regression was clearly the bigger harm: passes 5 through 8 captured groups at 4+ density far less often, and pass 9 restored that.

Now I'm checking whether black's drop lines up with the tengen opening: I'd compare black's win rate in games where we opened tengen against the ones where we didn't, across passes 6, 8, and 9 versus 4 and 5. If tengen is the cause, reverting it could bring black back to around 72%.

