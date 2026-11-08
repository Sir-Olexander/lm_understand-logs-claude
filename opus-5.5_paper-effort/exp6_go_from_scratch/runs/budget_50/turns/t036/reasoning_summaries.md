I'm checking the group at the top of the second example: five of our stones across rows 0 and 1, and I'm trying to see whether it still has a liberty at (0,6).

I'm checking whether that top group is really down to a single liberty at (0,6), and it is, so the group is in atari. Escaping at (0,6) wouldn't help either, since the new stone would still have only one liberty, so the group was already dead.

I'm working through the scoring for the two-liberty strings near the top-right, where my priorities aren't weighting the defense enough. Filling (0,7) looks safe on the count, but it would really leave only one liberty, so I'm thinking the defense bonus should only rise when a move actually adds liberties.

I'm working out how to gate the defense bonus so it only applies when a move gives the string at least three liberties, using a pair of units to express the threshold. The stumbling point is that the liberty count itself isn't directly available as a unit, so I'm trying to see whether it can be expressed as a non-negative quantity built from the surrounding stone counts.

I'm checking whether the gate can be built from the stone counts, and it turns out the liberty count is never negative, so the bounded terms are safe to use. But two units for the threshold would take 162 of my 107 spare units, so I'm stuck on how to fit it.

I'm weighing whether to make the defense bonus ungated and instead raise the penalty for two-liberty results, so that extending a two-liberty string only pays off when it reaches three or more liberties.

I'm weighing whether a flat penalty on two-liberty results could stand in for gating the defense bonus, though it would also punish ordinary contact plays and edge hane that produce two-liberty stones. I'm also considering whether ataris on opponent two-liberty strings are worth it, since they may just extend and gain nothing.

I'm settling on concrete values for the update: defense weight 7, the penalty for two-liberty results at 6, and the capture term weighted at 1. Since noise is about 5% over 96 games, I'll judge the change against the strong and weak opponents together before accepting it.

