I'm sorting out the self-atari cases in Go, starting with the one where the point has no empty neighbors and touches a two-liberty group. Playing there fills one of that group's own liberties, which is usually filling your own eye, and the rule is that this counts as self-atari unless two such groups are at a distance.

Now I'm checking the rule for the case where the point has one empty neighbor and only one of its own neighbor groups is in atari: playing there just leaves the group with a single liberty, so it stays unsafe. Then I'm working out how to fold the empty-neighbor count and the lowest liberty class of adjacent own groups into one unsafe test.

I'm working out a per-group liberty class so the unsafe test can be stated cleanly. For each own neighbor group, I need to know how many liberties it has, since that decides whether playing at p leaves the group in a bad spot.

I'm weighing whether a 3-coloring could make adjacent stones' liberty classes never share colors, since a 2-color scheme can't certify three liberties. Going through it, the pairing constraint between a stone and its neighbor looks impossible to satisfy with only three colors in general.

I'm checking whether a 3-coloring could work if even points all get one color and odd points get the other two, so that the liberties of a stone and of its neighboring group never share a color. Under that split, an odd stone's neighbors are all the same color, and an even stone's neighbors are drawn from the two odd colors. Next I'm looking at what the liberties of an even stone then look like.

I'm working out how each stone's liberties could be given color classes so that an odd stone's threshold counts can be certified with virtual colors. For an even stone, I'm now tallying how many of its neighboring liberties carry each of the two colors.

I'm checking whether a split like B/C for the even points could let the virtual colors cover the liberty counts for both stone types. The open question is how to set the threshold so a stone with two liberties in one color still counts as having two.

I'm working out the mod-3 claim for the even stone: B is claimed when it has a B liberty or at least two C liberties, and C in the mirror case. The A claim only turns on when there are three or more liberties in total.

I'm now working out how to assign the colors B and C to the odd points, since the even stones' liberties are drawn from those two colors. The aim is to keep the color classes spread out so that stones next to different even points don't end up with the same pattern.

I'm now looking at the mirror option, where the odd points take the one color and the even points split between the other two, which would give the same separation of neighbor colors.

I'm settling on the budget for each opponent and own-group setup: the opponent side uses a parity scheme with two units, and the own side uses the three-color version. I still need to test all three schemes against real positions at both radius 4 and 5.

I'm still working through the 3-color scheme for own groups, checking whether the threshold units can certify three liberty classes and where the linear readout breaks down. The plan is to use radius-4 NOT units for own stones and threshold units at radius 6 to pick out atari.

I'm checking when a point is unsafe depending on how many empty neighbors it has and the liberty count of its neighboring own groups. If it has two or more empty neighbors it's safe, and with one or zero I need the best neighboring group's liberty count to decide.

I'm checking whether a single threshold unit can express "all neighboring groups have at most one liberty" when each group's liberty count is only available as a linear function of the level-5 units, and that doesn't seem to work directly.

I'm trying to make the non-own groups drop out of the deficit sum, since their unit values would otherwise push the total positive by mistake. One option is to gate the unit values at non-own points so they contribute zero, and another is to use positive-polarity presence units at level 5, gated so they only count own stones.

I'm checking whether the own-group liberty counts can be summed into one linear quantity: using normalized presence units per neighbor and subtracting an own-stone indicator, so non-own points contribute zero and each own neighbor adds its present-liberty count minus one.

I'm working through how to make the normalized presence units zero out at non-own points, since the raw negation units give garbage there. I'm trying a gated positive-polarity version where each neighbor's count only counts when its point is own.

I'm checking whether non-own points can be forced to a fixed value by taking the NOT units from each point's own presence alone, without summing over neighbors. That way a non-own point gets exactly 1 and drops out of the deficit sum, so only own neighbors contribute.

I'm now trying a cleaner way to make the non-own points drop out: computing the NOT units at level 5 from each point's own presence only, with no propagation step, so non-own points stay at a fixed value. The other option I'm weighing is to keep the propagation step but gate it so only own neighbors feed in, though that runs into the problem of non-own points picking up neighbors' information.

I'm checking whether a single threshold unit can cover the radius-4 own-group case, or whether the non-own neighbors still leak garbage into the sum. Dropping the propagation step at the NOT layer would keep non-own points at a fixed value, but it costs one step of radius, so I'm weighing that trade-off.

I'm testing whether a per-stone "at least two liberties" unit at the next layer can feed a single unsafe check for the empty point, but that adds a layer I can't afford. I'm now looking at whether a linear output layer can approximate the same condition directly.

I'm trying to make a linear penalty stand in for the self-atari check: a term for when the point has no empty neighbors, a different term for one empty neighbor, and subtracted terms for neighboring groups with two or three liberties. I'm matching the weights so a neighbor group with three liberties cancels the penalty out.

I'm now trying to pool the neighbor contributions per point at the next layer, since adding each neighbor's penalty separately lets a point with several strong neighbors pick up a large bonus it shouldn't have.

I'm weighing whether to pool each empty point's neighbor colors at the next step, treating the point as a merged virtual stone. For an odd point its neighbors are all even, so the other liberty colors come from the neighbor groups and any empty neighbors, excluding the point's own color.

I'm weighing whether to pool the neighbor colors at each empty point as a merged virtual stone, and checking which colors remain once the point's own color is excluded. The open question is how claimed colors from a group that includes the point itself should count, so I'm working through that case.

I'm checking whether the pooled neighbor colors can be counted as lower bounds on the other liberties, since removing the point's own color from each group's claims keeps the count sound. The open worry is that virtual color claims from two different groups might double-count the same empty point.

I'm checking whether two groups that share the same liberty x could make the virtual color claims overcount, since the merged other-liberty set would then undercount the real libraries.

I'm checking whether two groups that both have virtual claims of the same color could still share a single real liberty, since that would make the merged count an overestimate. For odd stones the virtual claim always lands on B first, so their union stays small and sound, but for even stones I still need to work out whether the real colors and cross-virtual claims can collide.

I'm checking whether two neighbor groups of p can share a liberty and so double-count it, and noting that all neighbor stones of p have the same parity, which should keep their virtual claims from colliding.

I'm checking whether two neighbor groups of p can share a liberty and double-count it, since their virtual color claims might coincide. For the stones adjacent to p, the parity works out the same way, but a group that mixes odd and even stones could still break that, so I'm looking at what claims its real stones contribute.

I'm checking whether removing p's own color from a group's claimed colors still gives a valid lower bound on that group's other liberties. The worry is a group whose real liberties include p's color: if that color is dropped, the remaining claims should still undercount the other liberties, which keeps the check sound.

I'm checking whether a group's claimed colors can overcount when p's own color isn't among them. The worry is that a real liberty of the group might be p itself, and I need to confirm that p's color is always seeded as real from the stones adjacent to it.

I'm now checking whether the union of claimed colors across several groups can undercount or overcount real liberties, since a virtual color from one group might stand in for a different point than one from another. I'm trying to map each claimed color to a distinct liberty within each group so the total stays a sound lower bound.

I'm checking whether two groups that both touch p could share liberty x and overcount it. Since even-stone groups can each claim a virtual color, I'm working through whether their union of claims could still stand in for more liberties than really exist.

I'm checking whether two neighbor groups could double-count the same liberty, and it looks like the color-based claims stay consistent for stones adjacent to the same point. What's left is to test the merged check at p empirically, counting the neighbor groups' colors and the empty neighbors together.

I'm budgeting the layer-5 channels for the even-parity case: the board state, the O5 channels, the own-presence features, and the opposing parity-two channels together come to roughly 567 units. That overshoots the budget, so the opposing radius-5 channels can't simply be added in at this layer.

I'm now budgeting layer 5 again, with the opposing radius-4 channels and the own-presence features, to see whether the empty-neighbor counts can come from the opposing atari channels instead of the full O5 set.

I'm working out the even-parity case at p, where the neighbors are odd stones carrying two colors, one set vertically and one horizontally. That splits the empty-neighbor counts into two numbers, which breaks the single-count shortcut I had for odd p, so I'm checking whether the unit thresholds can absorb that split.

I'm checking the safety condition at p: it's safe when the two other colors are both present or when at least two empty neighbors exist, so it's unsafe only when there are at most one empty neighbor and one of the colors is missing. The catch is that for an even p, the empty neighbors split into two colors, so a single total count can't capture the condition.

I'm weighing whether the unsafe-check formula for even p can be written with only the total empty-neighbor count, since the two colors split the empties. One option I'm considering is reordering the coloring so the even-p case avoids needing the separate per-color counts.

I'm checking the option of dropping the opposing radius to 3, so the opposing atari channels can enter at layer 5 and fit the budget.

I'm working out the layer-6 budget now: the opposing atari channels can pass through at layer 5 or merge into the capture count, and the own-presence features at layer 4 leave room for the rest. The per-group escape check at layer 6 then needs a count of how many liberties each neighbor keeps.

I'm checking whether a per-color atari indicator can be written linearly at layer 6. Summing unnormalized presence values doesn't separate "two colors present" from "one color present with a large value," so I'm trying a per-color version that subtracts the other colors' presence with a bias term.

I'm budgeting layer 6 now: the board-accumulation units take 81, the liberty units take 162, and the per-neighbor atari checks take about 122, which leaves roughly 147 units for the eye and other patterns.

I'm checking whether dropping the opposing radius to 4 and using only the board channels at layer 5 keeps the budget under 486 units, but then the empty-neighbor counts for even points would be missing their per-color split.

I'm testing whether a monotone proxy from the neighbor's board value can stand in for the empty-neighbor count, since empty points read as 1, own stones as 2, and opponent stones as 0. The aim is to see whether "color B is available" can be captured without carrying the separate per-color empty counts.

I'm testing whether a neighbor that is my own stone can be told apart from an empty point, since an own neighbor shouldn't count as an available color. Empty and own points both read as positive, so I'm checking whether the own stones' presence values can separate them.

I'm trying to avoid storing n_e in the existing z slot by packing it alongside acc, since then reading acc back out would need n_e subtracted again. I'm weighing whether the liberty-bound logic can work with the group classes alone and skip the separate empty-neighbor count.

I'm checking whether the empty-point information for the opposing channels could be carried without leaking into neighboring cells. A signal at an empty point would propagate to its opposing neighbors at the next step, which breaks the design, unless the encoding stays below the threshold used in propagation.

I'm checking whether the empty-point garbage survives propagation: at an empty point the gate kills the carried value, so the unit would need a different gating there, and I'm working out the threshold that keeps that tiny value from leaking into opposing neighbors.

I'm weighing whether the even-parity empty-neighbor split can be carried by the layer-5 own-presence features instead, which would free units at layer 5 if those features stop at layer 4.

I'm weighing whether the own-side channels can drop to two colors per parity class, since the third channel only matters for the self-atari check when there are no empty neighbors.

I'm checking whether two channels can separate the "two other libs" case, since with only two parity classes the count of other liberties tops out at one, so the zero-empty-neighbor self-atari check can't be certified.

I'm weighing whether the opposing-color information can be carried at radius 3 with a parity-2 encoding, or whether a one-step lookahead at layer 1 could recover the missing step.

I'm checking whether a single linear unit can capture the "σ_n and at least one liberty elsewhere" condition for each neighbor, since a sum over neighbors can't tell an AND apart from its parts.

I'm checking whether a single relu can capture the case where an opposing neighbor is empty or the other color, since a false positive could come from the surrounding empty points pushing the sum past the threshold. A weighted version might work in most cases, but it wouldn't be exact, so I'm wary of relying on it. Using two units per point for the step-1 information would blow the budget, so I'm considering moving the opposing-atari detection to layer 6 instead.

I'm working through whether the layer-5 budget can drop the opposing-side features and still leave room for the layer-6 checks. The own-side features would stop at layer 4, but then the layer-6 liberty logic would still be short the empty-neighbor counts, and I'm trying to see whether an own-side virtual-stone step at layer 5 could supply them.

I'm looking at whether the own-side pres units at empty points can be repurposed to compute virtual-stone merges, by changing the gate so that empty points get a seed term instead of being zeroed out.

I'm checking whether the opposing-color gate can be made nonlinear enough to zero out empty points, but a linear gate through those three values is forced to be identically zero, so that route is out. I'm then looking at whether the opposing NOT units at layer 4 could help, and at carrying the empty-neighbor information inside the b unit's z slot instead of the accumulator.

I'm looking at carrying the empty-neighbor information inside each point's z slot at layer 6, where the eye-detection check is just a sum over neighbors. That works for odd parity, since a single count fits, but even-parity points would need two separate counts, which doesn't fit.

I'm trying to fit two separate even-parity counts into a single z value, using different scales so they can be separated. The problem is that each unit needs a different linear combination of the two counts, so one number can't serve both without the forms collapsing into each other.

I'm testing whether a single threshold on a combined value z = 3e_B + e_C can separate the two even-parity counts, since z ≤ 1 would then pin down e_B = 0 and e_C ≤ 1 at once. The next step is checking that the relu for U_B actually gives zero whenever e_B is 1 or more.

I'm trying to give each even-parity unit its own number, but b only carries one value per empty point. I'm wondering whether one of the opposing channels at an empty point could carry the second count without spilling into the opposing stones next to it.

I'm considering whether the opposing channels at empty points could hold the second count, but the gate that zeroes them at empty points would wipe out any carried value. A linear gate can't keep the value at empty points while zeroing it at opposing stones, so I'm checking whether the own-side channels at empty points could carry that second number instead.

I'm stuck on the even-parity points, where b can only hold one number and I need two separate counts. Giving one unit z = 3e_B + e_C would fit one check, but the second check needs a different combination, and the empty-point gate kills any value I try to park elsewhere. Maybe the coloring can be arranged so one class of neighbor colors behaves differently, which would remove the need for the second count.

I'm considering the even-parity case, where the horizontal and vertical neighbors take different colors. I could skip the color-merge entirely and use a simpler unsafe check based on each group's liberty class, but a product term over the empty-neighbor count and the group classes doesn't reduce to a single relu.

I'm checking whether two separate counts can be packed into one value by weighting one of them by 3 and the other by 1. A single threshold on that combined number would separate the cases, but only one unit can use it. The other unit's test is still unresolved, and I'm weighing whether a two-threshold split on n_e could cover both cases.

I'm checking the unsafe-move test by splitting it into two units: one that fires when there's exactly one empty neighbor and no own-side group has at least two liberties, and one that fires when there are no empty neighbors and no group has at least three. The sum of those two gives the overall penalty. To build both, I need per-group indicators for "at least two liberties" and "at least three liberties" as linear functions.

I'm checking whether the two-liberty count per group can be expressed as a linear threshold on normalized presence values, since the own-side test only needs a positive signal when at least two colors are present.

I'm checking whether the opposing channels can serve as a gate here, and the candidate form gives 0 at opposing stones but −1/2 at empty points, so it fails on the empty side.

I'm weighing whether normalized presence values, gated per own group, could replace the extra units, but that costs two units per group and breaks the layer budget. So I'm falling back to pure NOT units at layer 5, where a non-own group's count comes out exactly as 1.

I'm working out the sign for the non-own groups, where the constant offset has to cancel the count of neighbors that aren't mine.

I'm settling the budget: with the opposing radius at 3, layer 5 comes to 486 units, which fits. Own-side safety errors look like the bigger loss against a strong bot, since they outnumber the missed captures by a wide margin in the pass-1 stats.

I'm weighing whether the opponent-side false captures, where large groups with liberties beyond radius 3 get flagged, could be reduced by scaling the capture bonus with some proxy for group size, since I can't measure group size directly.

I'm weighing whether the false captures from big opponent groups with liberties beyond radius 3 could be cut back by scaling the capture bonus, though I can't measure group size directly. I'm also planning to measure how often the detector claims a capture at a point where no true capture exists, for radii 3, 4, and 5, and to check the own-side penalty for the at and U units.

I'm working through a way to pick out the relevant neighbor-liberty cases by propagating a signal at the first layer, aiming to combine the own-color and opposing-color checks into a single threshold.

I'm checking whether a neighbor-level term can filter out the non-matching neighbors, using a weighted sum so that stones of the other color push the value negative and only the matching color and empty liberties keep it positive.

I'm trying a different liberty test that reaches further, since a single OR-by-sum over neighbors breaks down: a negative term from a non-matching neighbor can cancel a positive one from a matching neighbor. So each AND has to be its own threshold unit, and I'm looking at whether propagating along straight runs could extend the reach.

I'm running the decision pipeline on the real data to compare configurations, checking how often each scheme gets a capture, an escape, or an unsafe move right.

I'm now writing the evaluation code: for each configuration, I'll compare the detected unsafe-move test against the true outcome, using both the color-set version and the simpler class-based bound.

