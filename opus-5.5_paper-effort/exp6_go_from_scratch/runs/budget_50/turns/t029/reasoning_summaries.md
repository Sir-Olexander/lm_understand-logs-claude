I'm noticing the opponent ends with noticeably more stones on the board, 45 to our 25. Since the scoring counts stones on the board plus empty territory, their extra stones and our passing both matter here. Looks like we lost a big chunk of the game.

I'm sorting out the liberty-based features now: classifying each string as having 1, 2, or 3+ liberties, then scoring captures, atari on the opponent, escapes, and self-atari penalties. I'm also working out how to compute the safety of a move by adding up liberty contributions from adjacent friendly strings.

I'm working out how to count a stone's liberties as 3 or more from local evidence: a stone with 3 liberties outright, or a stone with 2 liberties next to a same-colored stone with a disjoint liberty, or a stone with 1 liberty flanked by two opposite same-colored neighbors that each have liberties. I'll accept the rare error where perpendicular neighbors share a diagonal liberty and count it as 3+.

I'm working through a way to estimate whether a stone has three or more liberties by summing the liberties of itself and its neighbors, which can overcount when neighbors share a corner liberty. I'm accepting that a few cases will be wrongly certified, since the approximation is good enough for the features.

I'm working out how to certify a stone as having three or more liberties from local evidence, using a rough star count of the stone's liberties plus its same-colored neighbors' liberties. That gives an approximate class, and it can overcount when neighbors share a corner liberty, which I'm accepting. From there I'd split the two-liberty strings as those not certified at three or more but certified at two or more, and I'm budgeting the layered propagation for both colors.

I'm weighing whether a three-liberty certificate can be built from layer-1 values, since that would keep the layer-2 budget under control. Computing it directly from raw liberty counts would mean handling the color gating in a nonlinear way, so that path looks harder.

I'm running the memory budget for the layers: the second layer holds the stone, opponent, and evidence maps at 486 units with some room left, but putting the eye and accumulator in the third layer pushes it over, and the open question is whether the stone and opponent maps really need carrying through every layer.

I'm checking whether one shared propagation channel could serve both colors, since a neighbor should only pass on certification when it matches the stone's color. That color condition is nonlinear, so I'm weighing whether a relu-style gate can handle it within the layer budget.

I'm working out the memory budget again, since the second-layer counts for both colors plus the accumulator come to 567 units, which is 55 over the limit. I'm considering whether the stone and opponent maps really have to be carried in every layer, or whether each color's certificate and mask can be kept in a smaller set of units.

I'm dropping the accumulator from the middle layers and computing all the value terms at the final layer from the layer-5 maps, since the eye unit only needs the stone and opponent maps anyway. The output bias can cover the prior, so nothing is lost by skipping the accumulator in layers 2 through 5.

I'm settling the layout by dropping the accumulator from the middle layers and computing all the value terms once at the final layer, with the capture bonus carried in as a relu of summed opponent-propagation units. Each intermediate layer then holds the stone, opponent, propagation, and utility maps at 486 units, which fits the budget.

I'm laying out the last layer's feature units: an eye detector built from the stone and opponent maps, a capture-bonus unit summing the opponent's propagated certificates next to each point, and an atari-on-opponent unit counting adjacent opposing two-liberty stones.

I'm laying out the last layer as a set of feature units: a shared attack value that combines captures and atari on the opponent's two-liberty stones, and I'm now working on a safety score for each point that adds empty neighbors to the contribution of each friendly string, weighted by how many liberties it has.

I'm working out the penalty terms now: a self-atari unit that fires when a stone's post-move liberty score drops to 1 or below, and a weaker unit for a result of exactly 2 liberties. Then I'm moving on to a defense bonus for extending an own string that's in atari.

I'm running out of room in the per-point layout: eight units for each point already exceeds the 512 available in the last layer, and I'm trying to trim it down by merging or dropping some of the defense and penalty units.

I'm checking whether the scoring features can be squeezed into the last layer, and I'm trying to express the piecewise terms in s directly. Since s is never negative, a linear term in s can stand in for the relu(s-2) piece, leaving only relu(2-s) as an actual nonlinear unit to store.

I'm trimming the feature layout further: the stone-count term s can stand in for the relu(s-2) and relu(s-3) pieces, so only the relu(2-s) and relu(3-s) units need storing. I'm still over budget at seven units per point, and I'm considering whether the eye feature can be folded into the other units or whether it needs its own.

I'm dropping the separate escape-bonus gate and applying the escape bonus ungated instead, letting the penalty on low stone counts cancel it out. A multiplier of 25 on that penalty should handle the case with one adjacent atari stone, leaving the net near zero rather than positive.

I'm settling the positional features by folding every positive linear term into one unit per point, leaving the two penalty thresholds as their own units. That brings the last layer to about 324 units from those, with roughly 188 left over for the remaining positional terms, which I can split into a positive unit and a negative unit per point.

I'm settling the last layer into five per-point units: an eye detector, a positive linear unit, a negative linear unit, and two penalty units. That leaves about 107 units per point free for nonlinear positional features, like influence-based territory penalties. For the positional evaluation, I want to reward attacking opponent stones next to my point, weighted by how weak they are, and to reward extending my own weak strings.

I'm settling the positional terms now: a territory penalty for filling my own space, and a linear unit that rewards open empty points near my stones and near the opponent's.

I'm checking the eye and territory penalties against the pass logit, since moves in our own space or in the opponent's self-atari should lose to passing. Next I'm working out a local certificate for three or more liberties by summing a stone's liberties with those of its same-colored neighbors.

I'm checking whether the uncertainty propagation stays consistent with the certificates: a stone with a three-liberty certificate should also count as certified at two. But I'm worried the pseudo-counting could break that, since a point with no liberties and two neighbors with two liberties each might still fail the two-liberty check.

I'm checking whether a point with three or more liberties in its local star can really be certified at two, since a bad pseudo-count could break the chain between the two levels.

I'm checking whether the propagated uncertainty stays consistent with the certificates: a point certified at three should also be certified at two, which holds when a zero-liberty point with a two-liberty neighbor is covered by that neighbor's evidence. I still need to confirm the propagation step doesn't break that ordering.

I'm checking the tail of the uncertainty comparison: a point certified at three liberties could come out one short at two when the evidence sits at a farther radius, so the difference could be -1. I'm treating that rare case as a two-liberty stone in the score, and I'm also watching whether the attack term picks up a slight negative from it.

Still writing the layout of the per-point units...

Still writing out the per-point positional terms: contact with opponent stones, adjacent own stones, and diagonal support, with a term for open space nearby.

I'm setting the per-point weights now: capture and escape terms at 12 and 10, attack and defense at 4, eye penalty at 15, with the two liberty-threshold penalties at 25 and 3, and a small territory penalty.

I'm setting the penalty for moves that leave a string with only two liberties at 3, since those moves are often needed in contact fights. I'll write the build2.py file next.

