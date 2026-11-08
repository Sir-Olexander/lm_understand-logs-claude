I'm sorting out a few problems, starting with position 48, where a three-stone capture is outweighed by a penalty, so passing comes out ahead. I want the capture to dominate the penalties, since a capture should make the move safe.

I'm fixing the penalty logic so that a capture suppresses the penalties on the move, since a capture makes the move safe. Adjacent opponent stones in atari would be folded into the unsafe-move term, and I'm checking position 26 at (3,5) to see whether that suppression kills its penalty.

I'm checking the neighbours of (1,0): (0,0) is empty while (2,0) and (1,1) are white, so one liberty. My rough formula gives zero for U0 there, but I'm recomputing it since the number I wrote earlier was 6.

I'm checking the U0 calculation at (1,0) again, since the white neighbour at (2,0) is in atari and drags the value above zero even though one liberty remains. Making B0 larger would shrink that leak.

I'm working through the U0 term at (1,0) again. With n_e = 0 the merged group would have at most one liberty, so I'm trying to find a form where the atari neighbours don't leak a positive value when one liberty is still open.

I'm working out a way to stop the U0 term from leaking a positive value when one liberty remains. Dropping the atari negativity from the combined term leaves the two-liberty neighbour case at U0 = 4, and the mixed case with an atari neighbour would then come to 14 as a genuine self-atari.

I'm working out how big the penalty weight has to be so the no-liberty term stays near zero whenever an adjacent group is in atari. Holding the 0.5 term steady, I'd need the weight to reach about 80 before the positive leak is wiped out, which would push the 2-liberty values up to around 40 or 50.

I'm trying to find a form for the U0 term that stays positive only when n_e is zero. A large negative weight on the atari-neighbour sum would fix the leak, but the constants then pull in each other's bounds, and I can't pick one that works for both cases at once. So I'm looking for a formulation that avoids the negative weight altogether.

I'm trying to find a per-neighbour form for the U0 term, since no linear combination of the occupancy and mark counts gives a nonnegative quantity that is positive only when a group has three present stones. That forces a nonlinear term for each neighbouring group.

I'm trying to find a linear correction for the atari neighbours in the U0 term: something that equals 1 when a neighbouring group has exactly two of its three liberties filled, and that I can add to the negative weight so the leak goes away.

I'm still stuck on the U0 leak: the negative weight on the atari neighbours pushes a positive value through even when one liberty is open. Making the n_e term dominate would need a much larger base constant, but that inflates the two-liberty values, so I'm looking for a way to keep U0 confined to n_e = 0 without that trade-off.

I'm checking whether raising the base constant B0 to 24 with BIG0 at 10 keeps the U0 term at zero when one liberty remains and a single atari neighbour is present. That case works out, but if two or more atari stones touch the point, the leak comes back, and I'm still weighing whether that situation can actually arise here.

I'm settling on the U0 form with B0 = 24 and BIG0 = 12, which keeps the term at zero when one liberty remains next to an atari neighbour, and the mixed cases come out right.

I'm settling the U0 weights at 24 and 12 so the term stays near zero when one liberty remains, and I'm now looking at the U1 term and how large its weights need to be.

I'm working through the penalty weights for the self-atari and futile-extension cases, and settling on SA1 = 5 so that a futile extension nets a negative value.

I'm checking the strong-plus-atari case at n_e = 0, where U0 still comes out at 12 and the connection would be penalized more than it earns, so I'm weighing whether raising the escape weight to 20 flips the net sign without making the futile-extension case positive.

I'm settling the escape bonus against the futile-extension case, accepting that a connection to a strong group with no empty neighbours will sometimes be misjudged. Position 8 at (8,7) still comes out on top with the expected logit, and position 48 only needs its capture-suppression fix. Next I'll turn to the positional heuristic and the pass logic before the next games run.

I'm weighing whether to revisit the positional prior before the next games pass, since v1 strong games lost about 23 stones each and the new tactics may already fix much of that. I'm considering the cheap additions, starting with whether attacking opponent groups with two liberties can be detected from the existing features.

I'm considering a positional bonus for own groups with two liberties, since extending them would help defend against atari, and I'm checking whether a linear term over the existing features can express that.

I'm weighing a small defensive bonus for own two-liberty groups, since extending them only helps if it adds a liberty, so I'll set it to a light weight. For opponent groups, I'd like to press on their liberties, but the features can't tell me when they have only two, so that idea stalls for now. I'm now turning to the broader strategy question.

I'm checking the eye penalty against an escape: filling an eye to save an atari group currently scores as a loss, so I'm considering whether to suppress the eye term when an adjacent own group is in atari.

I'm working out the eye term now: I want it to drop when an adjacent own group is in atari, so that filling a point to save that group isn't scored as a loss.

I'm now checking the eye term: an atari neighbour should drop it, but a strong neighbour at the same time should keep it, so the sign of each neighbour's contribution matters.

I'm checking whether the eye term can ignore a neighbouring group that has only two liberties, since a living group with two eyes also sits at two liberties and filling one of them should still count as a real loss.

I'm looking at the eye penalty again: filling a false eye whose neighbouring group has only two liberties would be self-atari, and the existing U0 term already catches that case. Since a flat penalty means some useful false-eye connections will be missed, I'm keeping EYE at 40 without exceptions. Next I'll move on to the opening positional values, starting with the line weights.

I'm tuning the positional weights for the opening lines and the proximity terms, and I want to check that the defensive bonus fits in as well. Rather than adjust more, I'll apply the current fixes and run a games pass with 80 strong and 48 weak games, skipping random since it's already at full strength.

