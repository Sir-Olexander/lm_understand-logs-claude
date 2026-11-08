I'm checking the radius-1 counts for the n4 false positives, where I is own neighbours minus opponent neighbours of the stone. The true cases mostly sit at +1 or +2, but the false positives all seem to fall at zero or below, and I'm trying to work out why.

I'm weighing a hard veto at I ≤ 0, which would drop every false positive but also about 1,389 true atari cases, a sizable share of them big-group captures. Counts at each I level are roughly even between true and false, so any cutoff there would cut real captures too. I'm starting to lean toward giving up on a perfect split and considering a narrower suppression that only applies when I is at least 1.

I'm now considering a reversed version, where the suppression only applies when I is at least 1 and the capture bonus stays as it is. Big-group captures usually have strong bonuses that should outweigh the Uc penalty, though single-stone captures with I ≤ 0 might get pushed the wrong way.

I'm now working out how to make the suppression confident: the Uc penalty would only use a version of n4 that counts a stone only when its own-minus-opponent neighbour count is at least 1, so the I ≤ 0 cases drop out without a hard veto.

I'm working out a confident version of the suppression term, where each stone's contribution is adjusted by its neighbour balance so the I ≤ 0 cases drop out without a hard veto. The gating for own and empty stones looks like it stays negative, which is the behaviour I want.

I'm considering whether the veto can go into the bonus path instead. The catch is that n5 is built from n4, so scaling n4 by the I-count would also zero out the capture bonus for low-I captures, which I don't want.

I'm checking whether the layer-5 budget can absorb the I-count information, since replacing n4 with a scaled version would kill the capture bonus for low-I captures. I'm now looking at how many units each layer-5 channel uses, to see whether b5 could be dropped to make room.

I'm checking whether the positional term P2 can be computed directly from the layer-4 balances, since the difference of the two counts is just a linear sum of those balances. The remaining carrier term would then need the own and opponent totals separately, or some rearrangement of them, which is the part I still have to work out.

I'm checking whether I can get the I-count into layer 6 without losing the capture bonus, since rewriting the value as a function of A_o, I, and P2 leaves I still missing as a linear term.

I'm weighing whether to drop the v5 and n5 channels, which would free up 81 units at layer 5 and let the capture bonus fall back to the R4 signal. I'm still stuck on how to fit the I-count into the layer-6 value without losing the bonus for low-I captures.

I'm weighing a confident version of the suppression, n4c, that keeps the Uc penalty for self-ataris but drops the low-I capture cases, and I'm considering dropping the R5 channels to free the budget for it. The trade-off is unclear: R4 gives more false-positive bonuses, though most of those are harmless atari moves, and the real risk is the self-ataris, which n4c would still catch.

I'm now weighing a variant that drops the R5 channels, freeing 81 units at layer 5 for a confident suppression term, n4c. The bonus would then read from n4 instead, so more false-positive captures get rewarded. I'm still unsure whether the self-ataris caught by n4c outweigh that extra bonus.

I'm checking whether n5 can take the I-count from layer 5 at the cell q, without losing the capture bonus for low-I captures. I'm considering a version where the layer-6 term adds a correction that cancels the I-veto, so that n5 still reads as atari whether or not I is at least 1.

I'm considering whether to keep the bonus on the R5 channels and move the I-veto onto the suppression side instead, but that would need n4 both un-vetoed and vetoed at layer 5, which doesn't fit the budget. I'm now thinking about computing the vetoed suppression directly at layer 6 so that only one copy of n4 has to live at layer 5.

I'm looking at whether the I-based cutoff can go into the Uc term itself, as an additive correction instead of a multiplicative veto, so it only changes the result when the suppression is active.

I'm considering whether a term that depends on how many opponent neighbours a point has could do the suppression instead, adding a penalty only where the opponent dominates. That would leave non-capture moves and own-dominated captures alone, though it might also penalise some big-group captures in opponent-heavy areas.

I'm checking whether a distance-2 opponent count could serve as the suppression signal, since it would only add a penalty in opponent-dense areas. The worry is that for safe moves with a single neighbouring stone, the term might push a borderline score positive and penalise moves that are actually fine.

I'm weighing the last few candidate suppression terms, each one a trade-off between losing true captures and keeping the false-positive penalty. The distance-2 opponent-dominance idea is still borderline, since it could push some safe moves slightly positive. Before going further, I want to compare the options against the bigger gains I could get from pass decisions or move sharpness.

I'm now asking whether the false positives could be handled by a per-point nonlinearity, one that looks at the opponent-versus-own balance around each neighbouring point at layer 5. Without that per-point step, I keep running out of room, so I'm considering whether a cheaper, single-unit-per-point count could stand in for it.

I'm now looking at whether I can drop own-channel units by computing the m values at layer 4, but that would leave the suppression term at layer 5 depending on something it has not yet been given, so the dependency loops back on itself and doesn't close. Since this direction runs into the same budget wall, I'm going to step back and look for a better use of the effort.

I'm now comparing a variant that keeps the R4 bonus and uses the freed v5 slot for a confident suppression term, n4c, since the false positives mostly have an I-count at or below zero. The cost is that true captures in big opponent-heavy groups would lose their suppression and could then look like self-ataris.

I'm now weighing whether the I-veto should apply to the bonus at all, since big-group captures would be lost if it did. That points toward keeping the bonus unvetoed and using the suppression term only to block the self-atari cases.

I'm checking whether n5 can be built without n4 by ANDing over q and its neighbours, using the step-4 information already in the opponent neighbours' channels. That works for q with opponent neighbours, but an isolated q has no such cover, so I'm still stuck on that case.

I'm trying to see whether the layer-5 local liberty counts around q can stand in for the seed check, so that n5 could be built from units already available at that layer.

I'm trying to build a term that is 1 only when the point is an opponent stone with no own stone in its neighbourhood and at most one empty neighbour. The obvious way to encode the "at most one empty neighbour" condition is a penalty that scales with the empty-neighbour count, but a zero count would push the result above 1, so that version doesn't give a clean binary output.

I'm checking whether n_e(q) can be read from the neighbourhood counts, since a point adjacent to an empty point must have at least one liberty. That would let the term return 1 for n_e = 1 and drop to 0 at n_e = 2, though I still need to see whether the n_e = 0 case gets a value that breaks the binary output.

I'm checking whether the isolated-stone case works out: with no opponent neighbours, the R5 check reduces to whether the stone's seed lacks the parity, which is just the local liberty count. For stones with opponent neighbours, I still need to see whether their R5 set is covered by the neighbours' step-4 channels plus the stone's own step-3 information.

I'm checking the claim that the true-label set for q is covered by the neighbours' step-4 channels, since a stone with opponent neighbours has its step-4 information inside theirs. That would make the layer-5 check depend only on the parity counts there and on how many empty neighbours q has.

I'm seeing that n5 might not need n4 after all: if q has opponent neighbours, its step-4 information is already inside theirs, so the layer-5 check can run on the parity counts plus q's empty-neighbour count. The empty-neighbour term could be handled with a penalty that gives 1 only at exactly one liberty, but I still have to check how the zero-liberty case lands.

I'm counting the budget for the layer-5 units and coming up 17 over, since b5 for all 81 points, O5', m, v5, and the freed n4c slot plus the two globals exceed the limit. I'm now wondering whether the own-colour units on edge points could be trimmed to make room.

I'm now counting the layer-5 budget and finding it's still short, so I'm considering dropping the two global units or trimming the own-colour units. For the 17 skipped points, I'm wondering whether their parity and empty-neighbour information can be recovered from the neighbours' channels instead.

I'm now testing whether the skipped points can keep an n4 unit each, and whether dropping n4c from the 17 skipped points brings the layer-5 total down to 512.

I'm checking the 17 skipped edge points, where the suppression term isn't available. I'm wondering whether a true capture on one of those points could then get penalised as a self-atari, using an example opponent stone on the edge.

I'm checking that the suppression term is safe on the 64 points that have it, and that the 17 edge points without it only lose a bonus rather than triggering a false self-atari. I'm also working out the exact form of n4c: it penalises own-channel presence in the neighbourhood and adds the sum of the neighbouring opponent-count terms minus one, so full suppression needs at least one opponent neighbour.

I'm settling on the suppression term's behaviour: a true single-stone capture with two or more opponent-neighbour liberties keeps its suppression, while false positives with no such neighbours get none. For the bonus, I'm using the parity-and-empty-neighbour formula on 64 points and falling back to the n4 formula on the 17 edge points.

I'm writing out the n5 formula for the BSUB points, where the parity and empty-neighbour counts come from b5 and O5'. The open question is what to do with neighbours outside BSUB, since their empty-neighbour counts aren't available and treating them as zero would undercount n_e(q).

I'm now asking which points have skipped neighbours, since for those the neighbour's empty-count can't be read from the layer-5 units. If q touches a skipped point, the n5 formula would undercount its liberties, so I'd fall back to the n4-based version for those q.

I'm reconsidering the budget: if b5 covers all 81 points and n4c is kept only for 64 of them, the layer-5 total comes to 512. That would let the n5 formula run for every point without any n4 units, with suppression available on 64 of them. I still need to decide which 17 points to leave out of the n4c set.

I'm settling the budget by covering all 81 points with b5 and the layer-5 parity units, and keeping the n4c suppression unit only for the 64 points outside the skip set. That brings the layer-5 total to 512, and the layer-6 units come to 488.

I'm checking the gating on the n4c suppression unit for opponent points: with no own-channel presence the value stays at I_1, which is positive, but once the summed v4 terms reach one the margin shrinks, so a large I_1 could still leave it above zero.

I'm settling the layer-6 formula for the n5 unit, using the parity count and the empty-neighbour count from the layer-5 channels.

I'm checking the own-stone and empty-point cases of the layer-5 check, which both come out at zero, so the n5 unit is killed there as intended.

I'm settling the layer-6 suppression by making the veto stronger, so that a neighbouring stone's contribution can't push the unit above zero, and then checking the own-stone and empty-point cases, which both cancel to zero.

