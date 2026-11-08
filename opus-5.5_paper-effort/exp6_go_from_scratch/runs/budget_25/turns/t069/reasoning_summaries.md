I'm puzzled that (2,8) scores -0.3 here and ties with (1,4), with U0 showing 0, since I'd earlier worked out U0 for (2,8) as 11. I'm checking the count for the own stone at (3,8): its two liberties, (2,8) and (4,8), are both even-coloured, so it claims A and B, which suggests at least two present, and I'm now wondering whether propagation from (4,7) changes that.

I'm checking the neighbours of (3,8): both (2,8) and (4,8) are empty, so it's an isolated stone with colours A and B, which should give U0 = 11, yet my code returned 0. Tracing it through, the eye-merge term is 80·(1 − 3 + 1) = −80, and adding that to U0 zeroes it out for any point with fewer own neighbours than that. That looks like a real regression from v5 onward, since U0 is only effective when the point has deg − 1 own neighbours. I need an eye check and U0 that don't leak a negative term into non-eye points.

I'm seeing that the eye term turns negative for any point with two or more opponent neighbours, so it wipes out U0 for points like (2,8) that aren't eyes at all. I need to keep the eye bonus but stop it from dragging U0 down.

I'm weighing a separate unit for the eye check, as in v4, but that would push L6 from 488 to 569 units, which is too costly. Instead I'm trying to make the eye term non-negative by scaling it with the count of own neighbours, so that the penalty applies only to points with no empty neighbours and grows with their own-neighbour count.

I'm now considering whether a separate unit for the eye check would be cleaner, but that would push L6 past budget. Next I'm looking at whether the eye signal could go into U1 instead, though its negative term for strong groups would cause the same problem.

I'm checking whether folding the eye check into P2 would avoid the 81-unit cost at L6: P2 is a relu, so an eye-dependent term added inside it would only push the value up at eye points, and a negative output weight there would turn that into a penalty.

I'm checking whether the eye term could be made non-negative: adding it to P1 would drop P1 at non-eye points, and folding it into P2 would wreck the positional computation, so neither placement works cleanly.

I'm testing whether the eye term could instead be a linear penalty in P1, subtracting E times the own-neighbour count offset, but that flips the sign for non-eye points and gives them a bonus, which distorts the evaluation badly.

I'm looking at whether P1's linear part could move into P2 through the output weights, since P1 is effectively a linear term that only exists because the output can't read the layer-5 units directly.

I'm looking at whether P1's linear part can be expressed through relu units alone, since min(A_o, A_x) can be written as A_o minus a relu of their difference, but that still leaves a linear remainder I can't drop.

I'm working out whether the eye term can be avoided altogether: writing min(A_o, A_x) as (A_o + A_x − |A_o − A_x|)/2 would reduce the linear part to a single absolute-value difference, which would need two relus. Setting the weights so the sum term cancels would leave just a penalty on |A_o − A_x|, which rewards balanced points and changes how empty areas score compared with the current λ = 4.5, μ = 0.75 setup.

I'm weighing whether the escape bonus can be computed per point at this layer instead of spending 81 units on separate "at" units, since the linear carrier problem still has to be solved first.

I'm checking whether a max-based formulation could replace the linear part, and wondering if dropping the positional P2 term and relying on linear proximity alone is worth revisiting, since sharpening may have mattered more than the earlier 31% result suggested.

I'm checking whether the escape bonus could be folded into U1, but its negative term for strong groups would bring back the same issue, so I'm looking for other units at this layer that could absorb it.

I'm considering whether the layer-5 unit b4 could carry the A_x − A_o difference directly, so that P2 at layer 6 is just a relu of a precomputed value.

I'm trying to express the positional value through the signed difference of influences, A_x − A_o, plus their sum. The sum needs occupancy, which is nonlinear in b, so I'm checking whether a different form of the value only depends on the difference and A_o.

I'm now seeing that the linear part of the positional value can be rewritten in terms of A_o and the signed difference I = A_x − A_o, which means only I needs a relu. The catch is that I depends on the opponent's influence at L5, which I only have fully for the 64 sampled points, so the remaining 17 would have to be approximated.

I'm checking whether the linear carrier for P1 could be built from the signed influence difference I alone, using a precomputed relu at layer 5 from b4. Keeping P2 at layer 5 would cost 81 units, which is out of budget, so the b5 information would need to go instead.

I'm checking whether P1's linear part could be carried by a unit at layer 5 instead, using the signed difference I computed from b4. The catch is that A_o isn't available there, since occupancy is nonlinear in b4, so that placement doesn't work directly.

I'm weighing whether U1 and U0 could be folded into one unit with a combined penalty, since both depend on the number of empty neighbours, though their other terms differ enough that merging them would lose something.

I'm trying to fold the escape and eye-type penalties into a single relu unit: it should fire when there's one empty neighbour and the own-stone neighbours are all low on liberties, or when there are none and they're all at most two. The features I'd combine are the empty-neighbour count, the sum of (present − 1) across own stones, and the sum of (present − 2), and since the first two are tied together per stone, the combination has to respect that.

I'm trying a single relu unit that combines the empty-neighbour count with the own-stone liberty terms, and writing down the inequalities each case needs: an atari neighbour with one empty point should fire, a two-liberty neighbour with one empty point should not, and the same split holds with no empty points. I'm checking the cases with two empty neighbours and with no own neighbours too, to see whether one set of weights can satisfy all of them at once.

I'm checking the constraints on a, b and c for the single relu unit: a minus c has to be positive, and b has to sit between a/2 and a. Trying a = 4 and b = 3 works, leaving c between 1 and 3, so I'll take c = 2.

I'm checking the single-neighbour cases against the chosen weights: with the coefficients set, the case of one empty point next to a one-liberty own stone comes out positive, while the two-liberty and no-empty-point cases stay at or below zero. Next I'm testing the case of two empty neighbours with a one-liberty own stone, which should not fire.

I'm checking the multi-neighbour cases now: a strong group next to an atari one with no empty points comes out at zero, so it isn't penalised, which is an improvement. A two-liberty group joined to an atari one with no empty points does go positive and is penalised, as it should be. The remaining gap is two separate two-liberty stones with no empty points, which still score zero and slip through.

I'm checking the case of two stones from the same two-liberty group next to p, which might still leave only one liberty once p is filled, so the stone count alone doesn't capture it. I'm also testing one empty neighbour with two own stones: if both are in atari, the penalty should fire, but if one is a two-liberty stone, merging might leave two liberties and the penalty should not apply.

I'm settling on a single unit that penalises self-atari extensions, with the penalty weight set around 15 so that the extension and two-liberty cases come out at about 30.

I'm checking whether the k-term holds up when several own neighbours each carry strong groups, since then the sum of W stays high and k pushes the penalty down.

I'm checking whether a coefficient of 1 on the k-term rescues the case of three same-group two-liberty stones, since with 4 and 3 fixed, c = 1 would sit right at the boundary.

I'm checking the k=2 case where both own neighbours belong to the same two-liberty group, to see whether the reduced k-weight still catches a self-atari there.

I'm checking the penalty weights for the strong-plus-atari case with no empty neighbours, where the connection currently gets flagged as self-atari. With the k-coefficient at 1, that case should come out at about 30 instead of escaping, so I'm weighing how that trades against the same-group two-liberty cases.

I'm trying a c of 1.5 with c' at 2.5, which keeps the single-stone and two-liberty cases on the right side of zero, though the strong-plus-atari case without empty points only just clears the threshold.

I'm settling on a=4, b=3, c=1.5 and c'=2.5, with a self-atari penalty weight of 12, so that the single-stone, extension and two-liberty cases all land where they should.

I'm settling the penalty weights so the same-group cases and the strong-plus-atari connection both come out right, and I'm now checking whether the false-positive captures in games 1, 3, 74 and 104 are real captures rather than something the sim is miscounting.

I'm pulling the 1-liberty and 2-liberty cases apart: a self-atari extension costs about 30, while a (1,4) tie is still unexplained. I'm also planning to re-check the four flagged captures in games 1, 3, 74 and 104 for whether they're genuine.

I'm restoring the eye unit now, with the combined self-atari unit taking the place of the separate penalty terms.

