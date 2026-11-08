I'm going through the two cases: in the first, the capture gets flagged as a false positive because the group's liberties are far away. In the second, the check for C5 says it isn't atari, but C4 is treated as an atari case, which suppresses the self-atari penalty, and I'm wondering why (0,8) was picked over the top-scored move at (7,5).

I'm checking whether the (7,5) move with its 0.50 score was the one that should have been chosen over (0,8), which was sampled at lower probability. I'm also thinking about the false positive problem: big groups make the capture judgment unreliable, so I'm considering whether the capture bonus could be gated by group size, using the unnormalized walk counts in the L5 channels.

I'm working out a size-aware capture check: for a single stone in atari the walk count from its own color stays at 1, while a large group with many liberties produces a much bigger count, so I'm considering a threshold on that value to tell the two apart.

I'm checking whether a large group's walk count can gate the capture bonus, with the same-parity neighbor's seeds feeding into the count for the target point.

I'm weighing whether a walk count from the color-opposite seeds at the target point could serve as a size signal, since a lone stone in atari keeps a count of 1 while a large group produces a much bigger one. I'm considering whether a threshold around 100 would separate true atari from big-group false positives, though I'm not yet sure that number holds up.

I'm now weighing whether to avoid computing the walk-count values, since that would amount to reproducing the network's activations. I'm leaning toward semantic features instead, like group size or the number of liberties, and I'm considering a "big group" signal built from whether a stone has several same-colored neighbors.

I'm looking for a size signal that doesn't depend on counting walks, and I'm leaning toward a semantic check: the lone liberty of a true atari group should be surrounded by stones of the group's own parity, so I'm testing whether the opposing-color neighbors of that liberty can distinguish a small group from a sprawling one.

I'm trying to find a way to tell big groups from true atari without tallying walks, since that would mean reproducing the network's activations. I'm weighing a semantic signal instead, one that checks whether the group extends well beyond the liberty's immediate area.

I'm looking at whether the opposing-color neighbors of the liberty could serve as a size signal, and considering counting how many opponent stones sit next to the point: a point with two or more opposing neighbors would suggest a chain, while a true atari tends to have zero or one. That's a weak signal, though, since it only sees the immediate surroundings.

I'm considering whether a stronger signal could come from using both the R4 and R5 readouts together, or from the own-color side. The false positive count stays high even with an unbounded radius, since parity collisions alone produce hundreds of them, so the fix probably has to be semantic rather than a tighter radius.

I'm considering a further certification rule for stones of the opposite parity: if two such stones sit on a straight line with a same-parity stone between them, their liberties can't overlap, so the two-liberty evidence is solid. The L-shaped case is murkier, since diagonal neighbors share a corner point that might be their only liberty.

I'm considering whether a single-relu AND of two liberty checks can separate the straight-line case from the L-shaped one, but a single relu can't express that exactly, and the two-liberty situation is already covered by the single-stone certification.

I'm checking whether the opposing-color seeds can be spread without blowing the layer-one budget, since the straight-line certification would need around 126 units per side and only 26 are spare. Then I'm weighing which kind of error is costlier.

I'm considering whether the U-suppression could instead rely on R4NOT combined with the L5 values, since removing it outright would leave self-atari errors unchecked.

I'm checking whether the opposing-color seeds at the liberty could instead gate the U-suppression, but adding the unnormalized neighbor sums would raise the penalty for every move next to a healthy opposing group, which is the usual case. So that combination doesn't work as a simple sum.

I'm checking whether the opposing-color neighbors at the liberty could instead gate the suppression, but a single relu can't express the combination cleanly. Removing the R4 suppression altogether would leave self-atari errors unchecked, so I'm looking at a partial version: lowering the penalty so the suppression only partly applies. Next I want to see what the actual numbers say.

I'm wondering how much the false-positive captures actually cost, so I'll count pass 6 for moves where we played a capture-judged point that didn't capture anything. I can't see the network's own judgments in the records, but I can compute the semantic R5 judgment and check it against what happened.

