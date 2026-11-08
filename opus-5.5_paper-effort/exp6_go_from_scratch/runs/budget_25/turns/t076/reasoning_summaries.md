Most of the remaining self-atari cases are next to large opponent groups of six or more stones, and the model rates them as very likely captures (0.78 to 0.86). That points to the capture estimate on big groups as the source, since the bonus and suppression terms are pushing it wrong. The single-stone cases next to a big group account for 43 of them.

I'm weighing a way to spot big opponent groups cheaply, since the radius check can't tell whether a group continues past the search range. One idea is to seed the opponent color for stones with three or more same-colored neighbors, treating them as likely to have enough liberties, but that would misfire on real captures of large groups, which seem to be uncommon.

I'm working out a seed rule that counts a stone as a big-group signal when it has three or more same-colored neighbors, but the threshold has to be tuned so that two-neighbor stones don't trigger it too.

I'm working through the constraints on the weights: two same-colored neighbors should clear the threshold, while one same-colored plus two opponent neighbors should not. Checking the cases shows the three-opponent-neighbor case forces a contradiction with that, so a single linear seed won't separate them cleanly.

I'm checking whether any single linear seed can separate stones with two same-colored neighbors from stones with three opponent neighbors, and it can't, since the weights force a contradiction. So a different approach is needed, and I'm leaning toward limiting the false-positive damage through the suppression terms on the capture estimate instead.

I'm checking whether a group that runs past the search radius can be detected cheaply, since a group reaching beyond distance 4 would make the radius check unreliable. Propagating a reachability frontier outward from the stone would show whether any group stone sits at distance 5 or more.

I'm now considering a variant where every opponent stone gets a size-related seed on the opposite-parity channel, regardless of its liberty count. The idea is that this might carry a rough sense of group size without needing a separate propagation pass.

I'm checking whether a seed rule based on same-colored neighbor counts can separate the cases, but the weights contradict each other, so a single linear seed won't work. I'm now looking at the other option of making the opponent side use three colors, though that had a similar false-positive rate before. I'm also weighing whether propagating from the other side or using larger seeds at the first level could reach past the search radius.

I'm stuck on the same wall: a linear seed can't separate two-neighbor stones from three-opponent-neighbor stones, so I'm now trying to cut the cost of a large-group false positive by exploring a size proxy in the opposite-parity channel. Tallying the damage, roughly 1.1 self-ataris per game plus about 1.6 wasted moves per game, with the lost games carrying the heaviest share.

I'm considering a mitigation that lowers the capture bonus when the target stone touches several stones of the same atari group, since big groups tend to wrap around the point and those cases look like the false positives. True captures of a single stone would keep their full bonus.

I'm considering letting the suppression on the capture estimate weaken for these single-stone cases, since a true capture with no own neighbors would still net positive against the reduced penalty.

I'm settling on a compromise: a suppression of 2.5 per adjacent stone judged to be in atari. That brings the case of one captured stone with no own neighbors down to zero, and the single-stone case with one own neighbor to zero as well.

I'm checking whether the capture bonus or the suppression term should carry the fix, and the suppression numbers still come out to zero in both the big-group and single-stone false-positive cases. That means neither adjustment helps, so I'm turning back to finding a per-point group-size signal at the n5 level, possibly by also folding in how small the opposite-parity channel is at each neighbor.

I'm checking whether the group-size signal can come from the opposite-parity channel at each neighbor: if the walk count for that channel is large around a stone, that might indicate a big group, so I'm testing whether it should discount the n5 term for false-positive cases like game 9.

I'm estimating the walk counts for the group around (3,0), checking how many ways a 4-step walk with self-loops returns to it or reaches (2,1) through (3,1), since dense wall regions with degree around 3 should give counts in the tens.

I'm estimating how strong a walk-count signal would be for separating small true atari groups from the big wall groups, and it looks weak since the wall counts overlap with real 4-5 stone groups.

I'm considering whether a presence-based count over all group stones, rather than only the ones with liberties, would capture group size better, since walk counts would then grow with how big the connected group is. That separates tiny groups from large ones, but it needs an extra channel, and I'm checking whether the own-side channel budget can be trimmed to make room for it.

I'm now weighing whether a presence signal for group size could come from the opposite-parity seeds, but those seeds have to stay zero when the stone is absent, so that won't work. I'm falling back to accepting the false positives and looking for a different way to cut the self-atari damage.

I'm thinking about whether the big-group false positives can be detected through a per-neighbor check, asking if the opposite-color stones touching the move are also judged in atari. But I'd need group size information, and that's not cheaply available in this setup.

I'm now considering whether the layer structure can reach a radius of 6 for the opponent side, since the step-0 seeds are fixed at L1 and I'd have to see whether the own-side layers could be given up to make room. I'm also weighing whether a combined check at L1 could carry one more step of information.

I'm considering a different route to a larger effective radius: precomputing, at the first layer, a signal along straight lines of opponent stones (s, s+d, s+2d), so the wall's extent can be seen without adding more propagation steps.

I'm considering a discount on the capture bonus and suppression when the stone is surrounded by many opponent stones, since a dense opponent mass around the point is what separates the false positives. The catch is that the positional count isn't available as a usable unit at this layer, so multiplying it into the capture term would need a product I can't express. I'm then looking at adding a term to the suppression on the capture estimate instead.

I'm still weighing whether a discount on the capture term can come from the positional density around the point, adding a γ-weighted term to the suppression so that opponent-dense areas offset the penalty. The worry is that the same term would also push safe moves positive in those regions, so it only works if the safe-move margin stays clearly negative.

I'm weighing whether the empty neighbor of the move could tell a real capture from a false one, but I have no way yet to measure how big the adjacent opponent group is, so that check stays open.

I'm checking whether an empty neighbor of the move can separate a real capture from a false one, but the empty point is also a liberty of the wall's other side, so I'm reconsidering how to reach far enough across the board to see the group's real size.

I'm checking whether the 'interior' stones of a large false-positive group could be spotted, since a wall's libs usually sit at its ends or on its far side. I'm also wondering whether giving the opponent side a three-color scheme at radius 5 would help, though that budget isn't available right now. Next I want to look at the problem from the true-capture side.

I'm checking whether big opponent groups can be told apart by counting how many stones adjacent to the move are judged in atari, since true small captures usually touch just one. That would need a nonlinear bonus, which the per-neighbor weights can't express, so it doesn't fit as is.

I'm weighing whether a count of how many adjacent opponent stones are judged in atari could flag large groups, since true small captures usually touch just one, but a nonlinear bonus would be needed to use it. I'm also checking whether freeing room in the first layers, by trimming the b5 subset or the own m-units, could fund a size signal, though the spare capacity covers only a few points.

I'm working through how the own-side channels could be reorganized, since the even-point seeds seem to need all three colors and the reduction doesn't hold up. Using only two colors for the own side would break the three-class readout that the own-side capture check depends on, so I'm looking at what else can be trimmed.

I'm considering whether the escape signal could be folded into the existing unit budget, since the eye detection at L5 would also need a spot.

I'm checking whether the eye test can be moved to layer 5 by building it from the layer-4 values, which would avoid needing an extra unit there. Then I'd feed that result into the capture estimate as a non-negative term, so eye points get a positive push without any negative leakage.

I'm checking whether the eye test can be pushed into a single non-negative term over the layer-5 values, but a plain sum over the neighbors goes negative at ordinary points, so that version kills the other terms. I'm now looking for a combination that is zero everywhere except at true eyes.

I'm checking whether dropping radius 5 would free the 81 units at layer 5, but the radius-4 version would be weaker on false positives, and I'm weighing whether that freed space could carry a size signal for big opponent groups.

I'm looking for a local signal of how big the opponent group is, one that doesn't need more propagation. The idea is to count opponent stones near the point, since that needs a per-point opponent count, which the current layer can't express linearly.

I'm considering a density check inside the capture estimate: counting opponent stones within radius 2 of the point, which is linear at this layer through the previous layer's values. The idea is that a dense opponent area could veto the capture judgment, though the term might push the value above 1 at sparse points, which would need handling.

I'm trying to make the capture bonus fall off smoothly as the opponent-stone density around the point grows, so dense areas are vetoed without pushing sparse-area values above 1. The idea is to subtract a scaled density term and shift it by its minimum, so the value drops from 1 toward 0 as density rises, though intermediate densities would give partial values.

I'm checking whether the density of opponent stones within radius 2 separates the false-positive big groups from true small atari groups, so I'll measure that on the pass-10 data.

