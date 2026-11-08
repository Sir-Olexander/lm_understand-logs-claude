I'm trying to match a pattern of self-atari with two stones, but it doesn't line up exactly. Working through game 5, I'm tracking how white's capture of two stones at the edge leaves a single recapture, and checking whether the board after that matches an earlier position.

I'm checking the board positions around move 56 and seeing that black's stone at (0,1) plus the newly placed (0,0) made a two-stone group with only one liberty, so it was self-atari. White captured it, and the recapture at (0,1) brought the board back to an earlier state. I'm now asking why black would make that move at all, since the big white group nearby looked like it might already be short of liberties.

I'm weighing whether the (0,0) move was a false atari, since the (1,0) white stone has a liberty and its neighbors are already out of liberties. Working through the scoring terms, the self-atari penalty drops to zero only if that white stone counts as attacked, so the capture bonus is what's pulling the move in.

I'm checking whether the white group actually has real liberties. The stone at (1,0) has one, and the nearby stones at (3,0) and (3,1) look like isolated single-liberty stones too, so none of them are real seeds. The other liberties must be further away.

I'm checking whether the (1,0) stone is actually a false atari: its nearby stones are all single-liberty and spread out, so there's no real seed within reach. That would make the white group's apparent weakness an illusion, and the capture bonus would still outweigh the self-atari penalty.

I'm looking at game 11 now, where we play (0,8) and form a two-stone group with only one liberty at (2,8), so it's another self-atari that white can take and then I'd recapture into a repeated position.

I'm checking game 20 next, where a similar sequence seems to play out: a stone of ours at (8,8) gets captured and then the board state around it looks like it might repeat, though I'm not yet sure whether the position at move 84 matches move 79.

I'm comparing the board around game 20 to see whether the position at move 84 matches move 79, since white's recapture and our replay seem to be cycling the same stones back into place.

I'm noticing that the recurring pattern across these games is a self-atari of two stones, usually from a false atari or an escape attempt, followed by a recapture that the repetition rule punishes. The recapture itself is always taken because the capture bonus outweighs everything else.

I'm working out why the repetition hits: the recapture restores the board to a state from before the self-atari, because the stones they captured were exactly the ones we just placed.

I'm settling on the idea that the repetition comes from a self-atari of two stones, where we replay the same point after the capture. The real fix would be avoiding these self-ataris in the first place, which means getting better at spotting false ataris next to big opposing groups, though that's hard to do well.

I'm now considering whether the true-capture case behaves differently, since a real capture leaves the playing stone with liberties at the captured points. I'm thinking the self-atari kill weight should be tuned so that a capture bonus only wins when the capture is real.

I'm checking what happens to the self-atari penalty when we truly capture a dense group with no liberties of our own. The penalty would outweigh the capture bonus and make a real capture look bad, which seems wrong, so the kill weight needs to account for that case.

I'm weighing a fix that tunes the self-atari kill weight, but a dense false capture would still be penalized as if it were a real capture. Instead I'm looking at improving the reach that finds real seeds, using hashing to help the two-liberty cases.

I'm checking whether a hash-based check on the liberty positions would catch this group's second liberty at (4,0), since the two candidate points map to different classes and (4,0) is reachable from (1,0) within the search distance.

I'm weighing whether to drop the atari bonus for opponent stones, since the hash approach would cover the two-liberty cases but lose that signal. I'm going to quantify the hash variants on the dataset, focusing on the opponent captures that matter, to see how much the two-liberty failures actually drop.

I'm looking at whether I can widen the reach of the seed check directly in the h1 computation, using a linear threshold over a larger window instead of the product term, since that would catch the far-away liberties without needing hashing.

I'm checking whether the seed edges can fit in the h1 budget, since pairwise edge units for both colors plus single seeds come to about 612 units against a 512 limit. Splitting the edges across orientations or approximating per point would be ways to make it fit.

I'm trying to fit the edge seeds into the h1 budget, but folding the single-stone seeds into the edge units would need a version that fires without both stones being our own, and I'm checking whether a weighted sum of neighbor and own terms can do that.

I'm working out whether a single seed for a stone can be read from the first layer's signal, since the count of empty neighbors is linear in that signal. That would let the second layer combine edge seeds and single seeds into one unit.

I'm working out whether h2 can propagate edge seeds to neighboring points, so each point also sees the seed signals on edges touching its neighbors.

I'm noticing that edge seeds in the first layer give exact pair conditions, and that a second-layer hop can propagate them outward, which would push the reach to four steps. Adding a parallel set of edge units for the three-liberty check would be exact too, but it would cost another 288 units and blow the budget, so I'm looking for a cheaper way to encode it.

I'm working out whether the edge seeds can double as the three-liberty check, since an edge value of e(i)+e(j) would separate the two-liberty and three-liberty cases with thresholds. The catch is that a threshold on the sum is not linear, so I'd need the second layer to apply it in its pre-activation.

Now I'm testing whether a three-liberty flag can ride on the same edge units, by using a shifted threshold on the edge value, and I'm checking whether summing those per-edge flags in the second layer stays exact.

I'm checking whether summing the per-edge values can flag three liberties exactly. Two adjacent edges each sitting at a value of one could look like three liberties, but the stones' liberties might overlap, so the counting isn't clearly exact yet.

I'm now fitting the single-stone seed into the same second-layer unit, using the empty-neighbor count from the first layer's signal, so the edge and single checks can share one set of units.

I'm working out how to set up the pre-activation for each point's F unit, so that it fires when a stone has an incident edge seed, a neighbor's edge seed, or at least two empty neighbors, and cancels out for non-own points.

I'm checking the constant choices for the second-layer unit: a weight of 4 on the edge contributions, 1 on the single-stone term, and 2 on the neighbor propagation, so that an isolated stone with two empty neighbors still gets a positive pre-activation while a stone with none stays at zero.

I'm working out the pre-activation for the three-liberty unit F3 at each point, using a weighted sum over incident edge seeds with a threshold set to catch three or more liberties.

I'm working out the pre-activation for F3 by summing the edge values over each point's incident edges and subtracting the degree as a constant, so the offset can go into the bias.

I'm checking whether the three-liberty flag can use the edge values directly, by summing each point's incident edge contributions and offsetting by the degree so the constant goes into the bias.

I'm working out whether F3 can be computed in h2 from the edge values and the single-stone signal, but the isolated-stone case only fires at e(i) = 3 exactly, so the threshold still needs a check. I'm now deciding whether the budget leaves room for that or whether the approximation has to be loosened.

I'm checking the F3 pre-activation for an isolated stone with three liberties, where the threshold works out exactly. For a stone with one own neighbor, the case where the neighbor has three liberties and the stone has none gives a sum below the threshold, so this one miss has to be covered by the neighbor's own F3 unit.

I'm testing a weight choice for the three-liberty threshold: the edge coefficient a and the single-stone coefficient b against threshold t, checking each liberty case so that two-liberty configurations stay at or below zero and three-liberty ones come out positive.

I'm testing threshold values for the three-liberty unit, with the edge coefficient a and single-stone coefficient b, so that two-liberty cases stay at or below zero while three-liberty cases come out positive.

I'm testing whether a linear combination of edge value and single-stone count can separate two liberties from three. Trying weights a and b with threshold t: the case of two stones with one own neighbor needs a + 2b ≤ t, which is in tension with the three-liberty case.

Testing whether a single-unit three-liberty seed can be exact here: the constraints on the edge and single-stone weights seem to contradict each other, so I'm weighing whether to accept an approximation or to find room for extra edge units.

I'm checking whether a two-stone group at the edge with one stone already captured still produces a false three-liberty signal, since a stone with two liberties next to an own stone with none would push the pre-activation over the threshold. I'm accepting that as an approximation for now, since the budget can't spare the extra edge units needed for an exact check.

I'm reconsidering whether the old H-based formula for F3 fits the budget, since keeping H in the first layer would take too many units. I'm now checking whether the edge units alone can serve for the three-liberty seed, and what it would cost to keep them that way.

Still working through the three-liberty threshold for F3, and I'm weighing whether a seed from the edge values alone can stay exact or whether the approximation has to be accepted.

I'm testing whether a single edge-based seed for the three-liberty unit can be made exact, with a coefficient of 1 on the edge value and 1 on the stone's own count, against a threshold of 3.

I'm checking the three-liberty threshold against each case: the (1,2) pair comes out exactly at zero, so that miss gets picked up by the neighbor's own seed, and isolated stones only fire from e = 4 with this threshold.

I'm testing a weight of 1.2 on the single-stone term with a threshold of 3.4, checking whether isolated stones at three liberties clear it while the two-liberty pair cases stay below.

I'm checking a threshold of 3.4 with the edge weight at 1 and the single-stone weight at 1.2, case by case. The (1,2) configuration falls just short from the first stone's side, but the second stone's own seed catches it. The remaining cases in this pattern come out as expected.

I'm checking the two-own-neighbor case next: with the center at two and both neighbors at zero, each edge gives one, so the sum comes to 4.4, which clears the threshold and fires F3.

I'm noticing that the edge-based seed still gives false positives for a two-liberty stone next to an empty own neighbor, so the plain sum won't work. I'm considering subtracting a term tied to the number of own neighbors, which stays linear in the stone state and might fix the count without extra units.

I'm trying a linear form for the F3 seed where each edge contributes its excess over one liberty, and subtracting a term per own neighbor so that the count of own neighbors doesn't inflate the sum. The n_own·e_i term in the sum is the problem, since it grows with the stone's own liberty count rather than staying close to e_i plus the neighbors' values.

I'm weighing an alternative where per-edge units carry the combined liberty count of each own pair, but that would need per-edge thresholds the second layer can't express linearly. So I'm considering moving the three-liberty seeds to reach three, with the propagation work shifted to the other units.

Now I'm weighing whether to drop the exact F3 seed and accept the approximation, or to spend the remaining budget on it. I'm also considering whether the false case for a two-liberty stone with zero-liberty own neighbors matters more than keeping F reach intact.

I'm looking at whether edge values could give an exact three-liberty seed by using the excess over one liberty, which would make a pair with three total liberties fire while two-liberty pairs stay at zero. That would avoid needing the H units, though the first layer's budget is nearly full.

I'm working out whether an edge-based seed can cover the three-liberty case with a reach of 3, which would avoid the H units but cost some accuracy. Next I'll count the false-atari rates for the version with edge seeds at reach 4 and the three-liberty seed at reach 3.

