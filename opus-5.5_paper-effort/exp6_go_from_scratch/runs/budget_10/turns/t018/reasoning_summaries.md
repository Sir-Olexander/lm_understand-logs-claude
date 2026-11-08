I'm checking whether the position after move 79 repeats an earlier one, and it looks like it matches move 76 rather than 75.

I'm tracing the sequence of captures from move 76 onward: black takes two of our stones at 78, and when we recapture at 79 the board returns to its position from 76. That's a repetition under positional superko, so it counts as a loss even though the recapture itself is legal.

I'm checking the board positions at 74 against earlier moves, and it matches 71 rather than 72. So the 72 move, where we placed a stone that black then captured with two stones, was followed by black's capture and our recapture, which brings the board back to the position after 71.

I'm checking whether black's self-atari at ply 72 in game 92 was a deliberate move, testing the position after 71 where black's stone at (1,2) has only one liberty at (2,2) once we play (0,2).

I'm now checking whether black's stone at (1,2) in game 92 can escape, since it has two liberties, (0,2) and (2,2). I'm weighing whether playing (0,2) makes sense, and I'm not yet sure what drove the choice.

I'm working through the liberties of the white group around (0,1) and (2,0), which seem to have an open point at (0,2) and another at (3,0).

I'm looking at a possible false atari here: the stones at (0,1) and (2,0) each have only one liberty, and nothing links them to a seed, so the network may be reading them as threatened when they aren't. That would explain a +10 bonus that outweighs the -14 self-atari penalty on (0,2). I'm thinking the fix is better seeding, such as diagonal or straight-two patterns, so these false ataris stop distorting the move choice.

I'm working out why the group around (0,1) and (2,0) would be in false atari: those two stones are three apart, so they can't share a liberty, which means the group has at least two liberties and the atari signal is wrong. That suggests a rule for the seeding: if liberty-holding stones in one group sit at distance three or more, the group isn't really in atari.

I'm now thinking about whether a group in atari can be detected by tracking the positions of its liberty-holding stones. The trouble is that two distinct liberties can sit far apart, so a simple distance check doesn't settle it. I'm considering a coloring scheme, c(p) = (r + 2c) mod 5, as an alternative way to encode those positions.

I'm weighing a hash-based approach with a few flags per color class, since the flag budget per layer would allow four flags across 81 positions. The open question is which minimal set of flags would still catch most real atari cases, so I'm considering a single parity hash pair as an option.

I'm weighing whether two flags per color is the limit, and I'm considering one hash flag where the exact seeds carry the hash bit.

I'm considering whether a seed stone that has at least two liberties could just be marked with both hash flags, so the group reads as having two distinct liberties without needing a separate exact-seed flag. That would drop the third flag per color and keep the budget within four flags per layer.

I'm settling on two flags per color, where a seed stone with at least two liberties sets both flags, so a group counts as in atari only when its flags don't both fire. The open question is which hash of the liberty positions separates real 2-liberty groups best.

I'm considering a different kind of hash: instead of a fixed parity of coordinates, the liberty's direction relative to each liberty-holding stone, so that stones adjacent to a single shared liberty show a vertical or horizontal relation. I'm weighing how much that would separate real two-liberty groups from false atari, and which binary hash does that best on random positions.

I'm weighing whether a single absolute hash over liberty positions can stand in for the distance-based seeds, since a group in atari has all its liberty-holding stones within distance two of one point. I'm considering dropping the separate threshold flag and folding a stone's two-liberty status into both hash flags, which keeps two flags per color; the cost is that the two-liberty check (used for the atari bonus, extension, and self-atari estimate) gets lost.

I'm weighing whether a single two-flag hash can stand in for the separate seed flags, since that would free up budget but lose the two-liberty check. The alternative is to keep the exact seeds and improve them with a distance rule, though that might miss self-atari cases like game 92 ply 72, where a stone connects to a two-liberty group.

I'm considering a per-empty-point approach: for each empty point p, I'd track whether a neighboring group of a given color would still have a liberty if p were filled, which would detect captures at p directly.

I'm looking at whether a count of adjacencies between a group's stones and its liberties can tell me more than the distinct-liberty set alone, since a liberty touching several stones would show up in that count. That might give an exact atari test without needing extra flags per color.

I'm considering whether the carry of the S mask could be encoded in the value levels of each propagated unit, so that the mask and the flags share one budget.

I'm weighing whether to drop the threshold flag for one color so the budget still fits, since the two-liberty check feeds the atari bonus, extension, and self-atari estimate. Dropping it for just one color would keep both captures and escapes covered, though I'd be giving up part of the self-atari signal.

I'm now going to test the hash-based flags against the current seeded scheme on random positions and real game positions, to see which one catches more groups with two or more liberties.

I'm comparing a few encodings for the flags: the exact-seed-plus-distance option, a two-flag hash that drops the separate threshold flag, and a three-valued hash variant. The open question is whether count-based propagation could carry more information per unit than the binary flags do.

I'm weighing whether to keep the current two-flag design for both colors or split it, with hash flags for the opponent (to catch captures) and the existing flags for our own groups (for escapes, self-atari, and extension). I'm weighing that split by how much the lost two-liberty check costs for each side.

I'm weighing whether to drop the opponent's atari-move detection to keep the capture check, since the false opponent atari is what caused the repetition loss in game 92. I'm still undecided on whether the lost atari bonus against weak bots is worth keeping, and I'm leaning toward splitting the design so each color gets the flags it actually needs.

I'm looking at whether the budget could hold more flags if some of them were seeded later and used fewer propagation layers, which would leave room in the early layers. I'm also considering whether the stone-carry mask needs to appear in every layer or only some.

I'm planning to measure how often each candidate flag scheme falsely signals a capture on groups that actually have two or more liberties, using positions from the pass-2 games. For every group I'll compare the true liberty count against what the exact-seed-plus-distance option predicts, and against the hash-based options, then tally false-atari rates per adjacent opponent stone.

I'm writing out the seed rules for the two flag types, a local one based on each stone's own and neighboring liberties and an approximate one that sums liberties across the connected stones, then checking how far a seed's influence spreads through the group to decide whether an atari prediction holds.

I'm weighing a two-flag hash scheme against the current exact seeds, since the hash version would cover the atari check for every group, and I'm checking whether the F3 seeds ever misfire on true atari groups. They don't, so the hash option should be exact there too.

I'm weighing whether a binary hash over the board can tell distinct liberties of the same group apart, so I can get an exact atari test without the separate threshold flag. I'm wary of tuning the hash pattern on game positions, so I'd rather choose one on principle, aiming for distinct liberties of a group to rarely share a hash value.

I'm checking whether a binary hash over the board can separate distinct liberties of the same group, since a single hash function can't avoid all collisions. Simple parity patterns like checkerboard or row parity each fail for some distances, so I'm wondering whether any fixed rule can do well across the common cases.

I'm checking whether a binary hash can separate distance-2 pairs of liberties, since the simple parity patterns only handle some offsets. A block-based pattern like floor(r/2) plus floor(c/2) might split the (±2,0) and (0,±2) cases.

I'm comparing candidate hash patterns for the flags on real game positions to measure how often each one falsely signals a capture. I'm also wondering whether h1's spare units could support a larger seed window, since that might give more coverage without changing the layers.

I'm looking at whether a per-point helper in h1 could feed the seed formula in h2, since h1 has spare units to spend. The open question is whether a linear helper can express the local condition for isolated one-liberty stones, because an AND over neighbors isn't linear.

I'm checking whether a single relu can express an OR of several AND-type seed conditions, since the straight-two rule needs H at distance two plus a neighbor's own flag. Pre-activation offsets might let one unit encode the AND, which is the direction I'm testing now.

I'm working out a helper unit for the middle point of each straight triple, which would test whether both outer stones are ours and have liberties. The aim is to make the straight-two case an AND that a single linear term can express.

I'm checking the candidate helper that would test a middle point with both outer stones owned and their liberties summed to at least two, which would guarantee the group has two liberties. The ends can't share a liberty since their only common neighbor is the middle stone, which is already occupied, so the condition is sound for both the horizontal and vertical orientations.

I'm now looking at a helper unit per middle point that tests whether both outer stones are ours with liberties summing to at least two, but I'm worried about the unit budget since that's roughly 324 units and h1 only has 188 spare. I'm considering whether the H units themselves could be dropped or replaced by a cheaper per-point quantity, though a saturated version would need two units.

I'm now going to count how often each candidate hash scheme falsely flags groups that actually have two or more liberties, using positions from the pass-2 games.

I'm comparing the false-positive rate of the current flag scheme against the hash variants on pass-2 positions, counting every empty point next to a stone whose group actually has two or more liberties. I'll also check how often the T prediction matches for the current option.

