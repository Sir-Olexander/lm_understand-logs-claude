Most of the cycles seem to begin with a move on the first line, at edge or corner points like (0,4) or (8,8), which puts two or three stones into self-atari. They capture, we recapture one, and the position repeats. I'm looking at the throw-in shape where a stone at (0,5) has few liberties, and playing (0,4) puts two stones in atari.

I'm checking why this pattern gets probability 1.0, and suspect a false capture bonus on the first line: the opponent group's two liberties at (0,4) and (0,6) share the same parity, which seems to trigger a parity collision in the pattern check.

I'm weighing whether an edge-point shape where two of our liberties sit at (0,4) and (0,6) causes a parity collision that gives a false capture judgment on the first line. Playing (0,4) would leave our pair with a single liberty, but it would also leave the opponent group with one liberty, so the move's outcome depends on whether they recapture at (0,6).

I'm considering a targeted fix for the edge parity collision, perhaps a different two-coloring for the opponent side that flips colors on first-line points, so that pairs like (0,4) and (0,6) no longer share a parity.

I'm checking whether flipping the color of certain first-line points, like using a mod-4 offset on positions (0,4) and (0,6), would separate those edge liberties. Any coloring still preserves the lower-bound property, so the change only affects how well the detection works, not its correctness. I'm now working out how the seed assignment for par2 would need to change to match.

I'm working out the seed values for stones with mixed-color liberties: a color gets seeded if the stone has a liberty of that color, or if its total liberty count reaches two, which keeps the same threshold behavior as the even-parity case.

I'm working out whether the flipped colors on first-line points would still work for the NOT unit, since a neighbor's color might differ from the parity I assumed for that stone.

I'm checking the NOT units for first-line neighbors, where the flipped colors mean a stone might see mixed colors among its liberties. That would need two checks per affected point instead of one, which pushes the count past the L5 budget I'm trying to stay under.

I'm considering a different fix: handling the edge collision directly for libs two apart along the first line, since the two opponent stones above them share parity and so do their libs. I'm checking whether that same-parity pair is what causes the false judgment there.

I'm considering a targeted cert for the edge case: two opponent stones two apart on the second line, both with first-line liberties and joined through the middle stone, which would certify two liberties of the opposite color for the group.

I'm sketching a cert unit for the second-line triples: each opponent pair two apart along the line, joined through the middle stone, would give a virtual claim for the other color, using about 20 units at layer one. Those outputs would feed into the layer-two propagation as extra seeds at the middle stone's channel.

I'm settling on the claimed color for the cert: the libs at (0,c) and (0,c+2) carry the opposite parity to the 2nd-line stones, so the virtual claim should go to color par(s), which is what the middle stone's channel needs.

I'm extending the same triple-cert idea to corner cases: on the first line, a group like (1,0),(1,1),(1,2) with our stone at (0,1) between the two liberties would need the same treatment, giving seven triples per edge.

I'm weighing whether a cert over the 2nd-line triples would fit the budget, since the full set on both edges would need about 28 units against 22 spare. I'm considering trimming it to fewer triples, or limiting it to the edges where the same pattern shows up in the game logs. I'm also thinking about whether a more general fix for distance-2 collisions in straight lines would cover more cases than an edge-only patch.

I'm counting the edge triples on the second line: four edges with seven consecutive triples each gives 28 units, which is six more than the 22 I have spare at layer one. I'm checking whether the corner triples at c=0 and c=6 are worth keeping, since game 30 showed the corner case matters.

I'm checking the unit budget for the edge fix: the layer-one spare count is short by six, so I'm cutting the triples per edge down to about five. I'm also testing whether the cert unit, relu of three second-line stones plus both liberties minus ten, fires only when all three stones are present and both liberties are empty.

I'm adding a certification for the second-line case, where the cert fires only if three opponent stones sit in a row and both first-line liberties beneath the end stones are empty. I'm checking that it gives two distinct liberties and then sending its output into the layer-two propagation at the middle stone's channel. I'm also deciding which version to implement next, v13 or v15.

I'm settling on v13's parameters with the edge certification added, and I'm working out where the certification units go in the build: after the layer-one seeds, and as an extra term in the layer-two propagation for the middle stone's channel.

