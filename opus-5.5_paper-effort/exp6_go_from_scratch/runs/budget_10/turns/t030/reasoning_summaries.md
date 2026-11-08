The flat logit landscape in the interior is the problem: every point sits near 3.6, so the influence penalty barely bites since it only kicks in past 0.5 with a small weight. Right after black's move at (3,4), white's values are nearly uniform too, with even the neighboring points still at 3.6 because the contact bonus offsets the penalty. I need more contrast in the design, so I'm thinking about what the opening should reward.

I'm considering White's replies to Black's (3,4) stone, where the usual choices on 9x9 are a 3-3 point or a knight's-move approach. Since the goal is to avoid edges and wasted moves, I'll lean on standard Go heuristics for contrast: prefer lines 3 to 5, avoid crowding my own stones, and be careful about playing adjacent to the opponent unless there's a tactical reason. I'm also unsure what works best against this RL opponent, so I'm thinking about looking at how it opens as Black across its games.

I'm weighing an approach that scores moves by how much they shift territory ownership, using a clamped influence estimate so that contested points near the threshold matter most.

I'm working out how to make the contested-point score behave well once stones are on the board. In the opening every point counts as contested, so the score favors the center and third lines. Once areas near either side's stones are clearly claimed, those points stop counting, which pushes the choice toward open boundary areas.

I'm checking whether I can saturate the contested-point score without adding more units. Clamping needs two ReLUs per point for each of the two influence signals, which would push the layer to around 648 units, and that's too many.

I'm working out a contrast-based scoring rule: a clamped "tent" over the influence values that is largest where a point is still contested. That would make the score favor the center early, then shift toward open boundary areas once territory is claimed. I'm checking whether a two-unit combination can give that shape without adding more units to the layer.

I'm trying to get a scoring term that is largest where a point is still contested, which means its influence is near zero. A clean way to build that from ReLUs needs three units rather than two, and I'm weighing whether a looser penalty for strong own or opponent influence can approximate the same shape with fewer units.

I'm looking at why the penalty is too weak: with γ=0.3 and δ=0.2 it's small next to the base value of 0.3, so points deep in one side's area still come out around 3.7. The contact bonus adds to the flatness, so I'll need to check the influence values for a concrete position to see how strong the penalty really is.

I'm settling on explicit target values for the components: an open-area base near 3.6 at the center, and a penalty that should bring the score close to zero once a point sits about 1.5 units into one side's influence. With the penalty slope at 0.3, that gives a fairly modest penalty, and I'm checking what it does for points deeper in an area.

I'm checking whether the flatness actually matters: in the opening, many moves are about equally good, so the real risk is that the policy keeps sampling weak moves like edge points, which sit around 1.4 against roughly 3.6 in the center.

Edge and second-line moves would each take a sizable share of early samples, so I'm considering scaling the whole positional evaluation up by about 2.5 to sharpen the contrast. That would push edge and second-line weights down to near-negligible levels. I still need to check how the tactical bonuses like captures interact with that scaling.

I'm still weighing how to make the positional evaluation sharper, and I'm thinking about a tent-shaped contested-point score built from ReLU units. The unit budget is tight, so I'm checking whether a cheaper formulation can give the same contrast.

I'm checking whether a tent-shaped contested score fits in the unit budget, since the obvious three-ReLU form doesn't. The budget is already nearly full, so I'm looking at whether the owner flags could free up room for it.

I'm weighing whether the SA estimate can work with only the F flags, since a version that checks just e(p) and whether a neighboring own group has two or more liberties would miss some true self-atari cases. The earlier repetition came from connecting into a two-liberty group with zero empty neighbors, so the T_own flags seem worth keeping. I'm also considering whether the atari bonus still needs its own flags, since it does real work in the tactical scoring.

I'm considering whether the eye penalty can be folded into the existing self-atari unit instead of getting its own flag, since the unit budget is nearly full. The catch is that an eye point has zero empty neighbors but its surrounding own groups usually have plenty of liberties, so the self-atari term would switch off exactly where the eye penalty is needed.

I'm sketching a simple penalty for own-stone neighbors, where each own neighbor subtracts a fixed amount. But that can't tell an eye apart from an ordinary point, because edge points only have three neighbors. Normalizing by the number of neighbors softens the penalty too much to catch eyes sharply.

I'm weighing whether the unit budget allows the own-stone penalty to be expressed through the existing utility units, since adding a shared negative term to neighbor logits would need a sign flip that would also disturb the status bonuses.

I'm considering merging the eye penalty and the self-atari term into one combined unit, since they rarely overlap and the unit budget is nearly full. The overlap cases look harmless, so the combined version might work.

I'm weighing which six per-point features to keep, since the h6 layer is full. Dropping or merging features like the eye penalty and self-atari term keeps coming back to the same tradeoff: each merge either blunts the eye detection or switches off the self-atari check exactly where it's needed.

I'm weighing whether the tactical bonuses need rescaling along with the positional terms, and I'm picturing the opening landscape as a 3x3 high center, medium third line, and low second line and edge. Next I want to see whether a more distance-sensitive kernel can separate line three from the center.

I'm sorting out the opening priors on the 9x9 board: lines one and two get penalized, lines three through five stay neutral. I'm also weighing how the mid-game penalty should treat overconcentration near my own stones versus contact fights with the opponent, where tactics matter more.

Next I want to see how the strongest bot opens on both sides, so I'll pull up its first three moves as black and as white to get a sense of what good openings look like here.

