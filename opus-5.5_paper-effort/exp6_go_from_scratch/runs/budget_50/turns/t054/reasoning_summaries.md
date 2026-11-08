Big captures of ten or more stones show up often, and most of them come in games that were won, so they look like a driver of wins rather than a warning sign. That argues against distrusting large groups. They probably occur when the opponent's group is really dead, maybe after they've filled their own eyes. Meanwhile, self-atari blunders cost a lot over the sample, and I want to weigh that against the value of the big captures.

I'm weighing the alternative fix for the false-flag self-atari in the single_e0_libs[2] case, where the move has no empty neighbors and borders both an own two-liberty string and opponent stones flagged as in atari. If the flag is true, the move captures; if it's false, it's a self-atari, and I can't tell which from local information alone. To get more propagation reach for opponent strings I'd need to free up L1 units, and I'm now reworking the layout to see whether LS2 can be dropped.

I'm comparing a few ways to drop the L1 units: certifying strings with pairs as well as single stones, which comes out heavier at 612 units, or using pairs in only one direction, which would leave vertical strings uncovered. I'm weighing that against the single-stone certificates, which already handle a lot of the cases.

I'm now looking at whether a single unit could cover a whole "plus" shape around a point, not just one edge, so the pair certificates don't each need their own unit.

Now I'm dropping the explicit own-group evidence units and working out the opponent side directly: the opponent's evidence can come from the single-stone certificate plus the pair certificates, summed over each neighborhood.

I'm trying to drop one of the own-side certificate units, LS2, to get back under the budget of L1 units. One option is a replacement own certificate built from the single-stone counts alone, but I'm still checking whether it keeps the libs-at-least-two information I need.

I'm checking whether a single-stone check can replace the own-side evidence, with the pair certificates only added for the opponent side. The question is whether the reduced own certificate still keeps the at-least-two-liberties information I need.

I'm checking whether a cheaper certificate works without the LS2 unit: twice the string's own liberty count plus the summed liberty counts of its neighbors, with a threshold of 3. The opponent side can use the same form, and the reasoning holds since a neighbor with two liberties already implies a string of at least two.

I'm checking whether dropping LS2 and LO2 still works: the certificate 2·LS(q) + ΣLS(n) ≥ 3 holds in every case, including when the center has no liberties and a neighbor carries the count, so the own and opponent evidence can both come from single-stone counts. That frees 162 units and brings the layout back under budget. I'm now turning to how the ≥3 certificate handles the same cases.

Now I'm checking whether pair units can replace the single-stone opponent certificate, since the layout is still a unit over budget.

I'm settling on a version where the opponent's pair-based evidence starts the propagation at the second layer, and a separate single-stone certificate gets added at the third layer to cover isolated stones with two or more liberties.

I'm now checking whether the layout still works with the opponent's pair units and the second-layer evidence, which brings the count to 486. That leaves room for the third-layer propagation, but it means dropping the opponent three-liberty class, so the bonus for two-liberty opponent stones has to come from stone-level counts instead.

I'm checking whether dropping the opponent three-liberty class and carrying the opponent-side counts into layer two still fits the budget, since that version comes out at 567 units, which is over.

I'm trying to fold the own-side ≥2 and ≥3 certificates into a single unit, since both can be read off one combined count, which would save units on the layer where the budget is tight.

I'm trying to see whether a single unit can cover both the ≥2 and ≥3 conditions, since the ≥3 certificate implies the ≥2 one. A single integer-valued unit could capture the ≥2 case, but the ≥3 case needs a different threshold on a different sum, so one unit isn't enough by itself. I'm stuck on whether a derived unit can recover the ≥3 bit from the ≥2 one, and it doesn't look exact.

I'm considering a leaner fix instead: keep the current architecture and handle the false captures some other way. The false flags seem to come mostly from huge groups that are actually alive, and the pair approach only buys one extra step of propagation reach, which won't cover groups that span ten or more points.

I'm now testing whether a walk-based "reach" signal can tell how far a string extends, but walks can backtrack, so that signal is useless on its own. Non-backtracking would need direction information I don't have, so I'm considering a dead-end propagation to complement the certificates.

I'm weighing whether to tighten the trade-off by testing a smaller capture-credit setting, since with scap=1 a capture backed by two-liberty support still scores 2 and avoids the penalty, so that alone won't catch the problem case.

I'm weighing whether the own-side harm is even real: those 238 two-liberty self-atari stones sat next to a huge white group, so they may have been dead already, which would make the cost small. The repetition losses, about 5 of 128 games, are the part I want to pin down next.

I'm considering a pass-based fix for the repetition losses: when the board fills up late in the game, raising the pass tendency would stop us from wandering into dead areas. The pass logit could scale with the total stone count, which would need a stone-count unit in the last layer.

Now I'm weighing a pass-based fix for the repetition losses: raising the pass tendency with the stone count, which costs about one unit in the last layer. Real captures worth 12 or more would still beat that pass bonus, so only the low-value throw-ins in dead areas would give way. I'm also wondering whether the cycles could be targeted more narrowly at the ko-like captures themselves.

I'm going to check the repetition cycles by pulling up the final move of each of the seven games from pass 11, and for each one I'll count its empty neighbors, its own-side support liberties, and how many stones it captures.

