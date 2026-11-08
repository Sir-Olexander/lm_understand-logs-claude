The opening distribution looks flat, with top probabilities only around 3-6%, so the moves sampled early on are often weak. Game 18 shows a low-probability first-line move at ply 2, and the policy seems too uniform in quiet positions. I think sharpening the positional component's logit scale should let the good moves dominate, since the tactical parts already look sharp.

I'm weighing a sharper positional evaluation next, starting with scaling the positional logits up by about 2-3x. I'm also considering an influence-based approach where moves near contested points, where influence is balanced, score higher than ones in already-settled areas.

I'm working out how to get a per-point "contested" signal out of the network: a tent function of influence can be built from three ReLU units per point, and the output layer can then sum those over neighborhoods to score each move.

I'm now weighing a way to score moves by how contested the nearby points are. A value that ignores the sign of influence would push the search into opponent-held territory, so I'm looking at one-sided versions that stop at a plateau instead.

I'm working out a territory-style value for each move, where a move's worth comes from the nearby points' estimated ownership, with those already firmly opponent-held counting less. I'm also weighing whether an invasion into opponent territory is worth it on 9x9 and whether a cap or penalty should limit it.

I'm settling on a per-point influence feature: one unit per empty point that is large where influence is neutral or opponent-held and small where it is already ours. A second unit subtracts a term for deep opponent areas, so the value plateaus instead of pulling the search into them. Tactical and shape considerations like contact, connecting, and cutting still need to be handled separately.

Next I'm weighing whether the opponents' rating spread suggests the stronger bots need the most attention. The priorities come out as sharpening the positional policy, then improving the influence-based evaluation, then fixing false captures, and finally checking how games end, since losses are close.

I'm checking whether the endgame costs us the close losses: passing too early while dame or boundary points were still open, and failing to capture dead stones. Next I'll look at the final boards and last moves of games 36 and 29, both lost by 0.5, to see if we passed too soon.

