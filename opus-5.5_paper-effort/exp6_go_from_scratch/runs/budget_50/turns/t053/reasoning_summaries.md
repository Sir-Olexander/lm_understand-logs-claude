I'm weighing the move by how many stones it would lose. The biggest case is the single stone at the top, which is deliberate: it has no empty neighbors and touches one of my own strings with two liberties, so the score comes out to a -25 penalty. It still gets picked more than half the time, which suggests a large bonus from a false atari flag on the opponent's neighbors, adding 2 to the score.

I'm working through the false capture flags on big opponent groups, since a flagged atari gives an inflated score and removes the penalty. Next I need to find out when such a flag is actually false, and whether a size bound on the string could be used to trust captures only where they're verified.

I'm considering whether a cheap density check could stand in for an exact string-size test, counting opponent stones near the point to flag likely large groups. I'm also thinking about moving the local certificate computation into the first layer so propagation gets one more step.

I'm looking at whether the local evidence check can be split into separate first-layer units, one for a single opponent stone with at least two liberties and one for a pair of adjacent opponent stones whose combined liberties reach two. If each piece is a simple threshold on the raw inputs, propagation could start a layer earlier.

I'm working out whether the pair units can stand in for the triple case, which I'm leaning toward skipping since the neighbors' own liberties aren't covered. The next step is tallying the L1 budget across the single, pair and liberty-count units to see whether it still fits.

I'm tallying the L1 budget and it comes to 612 units, over the 512 limit. Dropping the independent-stone units doesn't help, since the total is still 612 either way. I'm now asking whether the single-stone certificate units can be folded into the pair units, so I don't need them separately.

I'm tallying the L1 budget again: the stone-count and occupancy units take 162, the liberty-certificate units another 162, and the pair units 288, which brings it to 612 against the 512 limit.

I'm checking whether dropping the ≥3 class for opponent strings frees enough L1 budget, since the total is still around 549 against the 512 limit.

I'm estimating the L1 budget with a cheaper option: keep the opponent pair units and the own-string certificates, and drop the opponent three-stone class. Even so, the layer comes to 549 against the 512 limit, so I'm still 37 over.

I'm trying to get rid of the separate own-string units by computing the own-string evidence another way, but that doesn't work. I'm now considering merging the O carry with the A certificate into one unit, since the layer still needs O on its own for masking.

I'm considering dropping the own-string three-liberty class as well, but then a healthy own group with three or more liberties would count as only one point of safety. That could wrongly flag a move next to it as self-atari, so I'm weighing whether that false positive is acceptable.

I'm weighing a cheaper fix for the false big-group flags, since a density check or a lower scap value both seem to cost too much. Lowering scap to 0 cut captures per game to 8, and scap 1 gave 37.5% but was confounded by the other changes.

I'm checking whether a capture with scap=1 still gets through, since a single empty neighbor with no own support would give s=2 and skip the penalty. That means the false capture flags on a big opponent group would still be chosen.

I'm considering a distrust heuristic that gates the capture score by local smallness, counting opponent stones within two steps of the point. The open question is how many of our captures are large ones, six or more stones, since those are the ones this would cost us, so I want to tally the capture sizes from the earlier results.

I'm checking whether distrusting captures on big groups would cost us real captures, so I'm trying to see how often our captures are large ones.

