I'm looking at a Go position with my top group around (1,4) and (2,2) through (4,2), now surrounded by the opponent's stones along the edge. I played elsewhere while that group ran out of liberties, and wonder whether my moves at the bottom were aimed at territory.

I'm weighing whether a "weak group" feature is worth the cost: a large string with only 3-4 liberties should be pushed toward open space, but my defense only tracks strings with 1-2 liberties, so a third class would need another propagation pass. The budget looks too tight, so I'm considering whether the opponent's ≥3-liberty class could be reused for attack instead.

I'm considering freeing budget by reusing the unit channels in the middle layers, since a per-neighbor weakness signal would need another layer that doesn't fit.

I'm wondering whether I can merge the S and O channels into a single signal, using values like 2 for my stones and 1 for the opponent's, so I don't need separate units for each. The catch is that I'd still need to pull out an exact mask for my own stones during propagation, and I'm not sure a linear readout from that combined value would give me that.

I'm checking whether the own-stone mask can be read out of a single combined value by thresholding it, since subtracting the constant 1 would separate the two cases. The sign of the opponent term is the snag: a constant offset that works for my stones leaves the opponent's stones and empty points in an ambiguous range.

I'm checking whether a single signal can separate empty points from the opponent's stones, since a one-unit encoding for own and opponent stones would leave the mask ambiguous. Pairing two units seems to be the minimum for three states, so I'm testing whether combining them with the attack-count units could free space without losing the own-stone mask.

I'm considering dropping the ≥3-liberty classes to free about 162 units per layer, but they feed the defensive and offensive bonuses and the self-atari check, so losing them would blur the difference between 2-liberty and safer strings.

I'm considering whether the current design can afford a weak-group signal at all, and leaning toward tuning the existing parameters for group safety instead. I also want to revisit a linear way to reward moves that extend a thin group into open space.

I'm wondering whether our groups die mainly because they get enclosed, since the opponent plays at temperature 1 and the bots aren't perfect. I'm considering whether moves that keep groups connected to open space could help, though without lookahead that's hard to judge. I also want to tune the remaining passes, focusing on the parameters that affect group safety.

I'm running a set of parameter experiments next, starting with raising the defense weight to 6 and the M penalty to 8, then testing the open-space term, the attack weight, and the pass settings.

