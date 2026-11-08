I'm sorting the failures by distance, and the 2-lib cases are mostly ones with no seed at all: 263 of them. The 22, 1, and 1 at distances 4 to 8 come to 24 for that group, while the ≥3-lib groups have 469 at distances 4 to 8 and 99 in the "9" bucket.

I'm working out how many failures could be fixed with one or two extra hops of reach: about 221 of the 855 for the ≥3-lib groups with one more hop, and roughly 330 with two. For the 2-lib no-seed cases, a mod-5 hash would recover around 100 of the 263.

I'm weighing whether to cut the damage from false atari instead of chasing perfect detection. A capture bonus that is added per neighbouring stone could be paid out twice on one fake signal, which would outweigh the self-atari penalty in a game like 92 at ply 72. If the capture bonus were counted only once, that move would net negative and wouldn't be played.

I'm now considering whether a capture bonus could be made non-additive, so a single fake signal can't pay out more than once. A plain relu can't saturate that way, so I'm looking at whether the self-atari penalty should still apply even when a capture is detected.

I'm weighing whether reach or false-atari damage is the better thing to attack. The worry is that a capture bonus paid per neighbouring stone could double-count one fake signal and outweigh the self-atari penalty, which is what happened in game 92. Options for extending reach without giving up detection are still open.

Still working through the segment-based propagation idea for reaching longer straight chains.

I'm now thinking through whether segment units of length 3, horizontal and vertical, could carry information along straight runs in a single step, though that would only save time if the segment values were usable in the same layer.

I'm looking at whether segment seeds of straight runs, computed directly from the inputs in the first layer, could extend reach for long chains without relying on per-point propagation. The count of horizontal segments of each length across the board is what I'm sizing up next.

I'm looking at whether doubling the reach of each propagation step could cover more of the long straight chains without spending the remaining budget on more segment seeds.

I'm considering whether to accept the remaining false-atari damage and reduce its impact, since per-point propagation can't cheaply reach long straight groups without mask products that break linearity.

I'm considering making the capture bonus saturate instead of summing across neighbouring stones, and checking whether self-atari should still apply when a capture is detected, since true captures free up liberties while fake ones don't.

I'm checking whether a capture bonus can coexist with the self-atari penalty, using a partial penalty so a single atari neighbour doesn't wipe out the capture signal. Then I'm considering a local test: if a neighbouring stone belongs to a group in atari, every other liberty-holding stone in that group should sit next to the capture point, so any stone that still has a liberty elsewhere would reveal a false signal.

I'm now weighing a restructuring where the saturating step moves into the final feature layer, which would free up one propagation hop. The catch is that the self-atari and eye features need the saturated liberty flags, so they'd have to be rebuilt another way.

I'm splitting the saturated A and T flags into separate units for each opponent and own color, so each one stays a clean relu over the unsaturated hop-5 flags. Then the self-atari count can use the own-group three-liberty flags directly, with a kill term of -2 per such neighbour.

I'm checking whether the self-atari term can be built from the saturated-free flags, so the penalty for own groups with three or more liberties stays linear. The reach mismatch between the three-liberty count and the A/T flags is small enough to accept, and I'm dropping the capture-dependent kill term in favour of relying on the capture bonus outweighing the penalty.

I'm now checking whether the EYE term can be folded into the SA feature, since both are self-only checks on each point. Each unit in the final layer is already linear, so the remaining question is whether J, OT, XT, and NST can share units without going over budget.

I'm looking at whether the eye check can be merged into a single weighted unit, with the own-neighbour count weighted heavily and an opponent penalty in the wider ring. The concern is that eyes sitting next to opponent stones could still slip past that weighting.

I'm checking whether J could ride on an existing unit: since the output layer is linear, a linear board term could come free if it were folded into a unit that is always positive, but that would still need its own slot unless it can be merged into one of the others.

I'm now trying to cut the A and T indicators down to fewer units per point, perhaps by merging the opponent and own versions into a single relu, though the cross-terms would have to cancel cleanly.

I'm checking whether the h6 budget can be saved by merging the A and T indicators, but any combined unit would pick up flags from the opposite colour and wrongly suppress them. Making h2 a propagation layer would need exact pair seeds from a two-layer computation, so I'm leaning toward keeping the current layout and accepting reach of 3.

I'm weighing whether the second layer can compute pair seeds directly, using 162 extra units in the first layer for simple single-stone seeds, which keeps the total within budget.

I'm checking whether the e≥2 seeds can gain a hop, and the pair seeds still need a mask weight M on the stone's own indicator so non-own points can't pick up stray neighbour counts.

I'm checking whether the F3 seed can come from the first layer's e-minus-one values, since a neighbour with three or more liberties shows up as a value of at least two there, though that threshold is awkward to express linearly.

I'm checking whether the seeds at reach 4 could help the long straight groups, since only the e≥2 seeds would gain that extra hop while the F3 seeds stay at reach 3.

I'm weighing whether a segment unit could spread the frontier signal along straight runs in a single layer, which would only pay off if the next layer can still distribute the saturated flags.

I'm now thinking about whether a single layer could move a signal further along a straight line, with each point reading the flags of a segment of neighbours in one step rather than passing them one hop at a time.

I'm now considering whether the propagation can be spread across several units per point, each covering a different path shape with its own mask, since the next layer can simply sum them to get an OR.

I'm working through the budget again: each point would need four separate units per flag per colour for straight-line propagation, which is far too many, so I'm checking what the total unit count still allows.

I'm considering whether the budget could cover extra reach by adding two-hop units in straight lines, but any unit that mixes both directions would need masks on both middle points, which is too strict. I'm now thinking the better route is to compute seeds from the first layer using exact rules over slightly larger local shapes, since that uses the spare units there.

I'm weighing whether the capture bonus should stay moderate against the self-atari penalty, since a true capture is rarely self-atari, though a capture that leaves the played stone with only the freed points as liberties could still be.

I'm weighing a hybrid: keep saturation for the A and T flags at reach 3, while propagating F alone one more hop to reach 4, then feed that reach-4 signal into the next layer as an extra kill term.

I'm considering a sixth-layer unit that subtracts the propagated opponent-frontier count from the saturated atari signal, which would push the effective reach to four, though the summed term is still unfinished in my notes.

I'm checking whether the opponent-frontier term can be split into its own unit, so the kill signal from neighbouring opponent stones doesn't also wipe out the own-stone contributions.

I'm running the unit budget for the h6 layer: a separate capture unit and escape unit for each colour, a combined unit for the two-liberty terms, plus eye, self-atari, jump, and the other terms. That totals 648, which is over budget, so I'm considering dropping the OT and XT units or merging them into one.

I'm sizing up how the h5 and h6 layers would fit in the unit budget: the saturated A/T flags, the propagated F signal, and the S units together would total 648, which is over. Structural savings are going to be hard to find, since both colours' S and the T units feed directly into the masks and the final score.

I'm checking whether the constant part of the prior could ride along inside the stone-status units instead of needing its own J unit, since the output sums those units over each point's neighbours.

I'm looking at whether the per-point self features like EYE and SA could share a unit, since they rarely fire together. I'm also considering folding the constant board terms into the stone-status units, though the distance weighting would leak status bonuses onto diagonal points.

I'm weighing whether a penalty unit that combines the own-neighbour count with the eye condition could stand in for the separate EYE and SA terms, though the exact shape is still messy. Before going further on the architecture, I want to know what is actually driving the losses against the strong opponent, so I'm planning to look at a few lost games and sort them by failure mode, such as group deaths, territory deficits, false captures, eye-filling, or passing too early.

