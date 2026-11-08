I'm going through the board after the pass, checking which black stones along the top-right look dead and which points sit inside white's territory.

I'm checking the stones near the edges that look like they belong to the other side, and working out whether they count as dead.

I'm working out whether we should keep playing when behind instead of passing, which means estimating the score from stone counts and rough territory before deciding to pass. If we're behind, invading the opponent's area or filling neutral points makes more sense than passing.

Playing inside the opponent's territory costs nothing under area scoring, since captured stones don't count, so it's a free way to threaten and possibly catch them out. Stones that survive to the move limit still count for me, so late invasions can break their territory. When I'm ahead, passing ends the game, but when I'm behind I should never pass, because the opponent passing would end it with me losing.

I'm sketching a rough score estimate: each stone adds or subtracts a point, and empty points touching only one side's stones count for that side. Then I'm checking how well that proxy works, especially in open regions where there are no nearby stones to go on.

I'm working out a per-point score proxy, where an empty point counts for a side only if its neighbors are mostly that side's stones and the opponent has none.

I'm working on a score proxy where each empty point counts for a side based on how many neighboring stones belong to that side versus the other, with the weighting capped so one stray stone doesn't dominate.

I'm checking how a score proxy would count territory, and I'm stuck on the fact that the unit count doesn't fit: the pass decision sits in the output layer, so any territory units would have to live in the last layer, which only has 26 spare slots.

I'm checking whether a single scalar score estimate could be computed in the first layer, with its territory indicators, and then carried forward through the later layers to feed the pass logit.

I'm settling on how komi enters the score: a white-to-move indicator in the first layer, shifting the estimate by plus or minus 7.5 depending on which side we're playing.

I'm adding the territory terms in as weighted sums, with the komi shift and an offset to keep the score unit positive, and the pass bias then subtracts that offset. The stone counts come straight from the first layer, so the only real question left is how to make the per-point territory unit closer to binary.

I'm sizing the territory term, weighting each empty point by how many neighbouring stones belong to one side, with a scale factor to keep it in range. I'm noting that stones I've already counted, including dead ones, are right to count as they are if the game ended now, while large empty interior areas will be underestimated because they have no adjacent stones. The open question is whether the pass logit should still favour continuing when the estimated margin is positive but capture chances remain.

I'm weighing whether a pass logit driven by the estimated margin would trigger too early mid-game, since an ahead estimate could push passing above ordinary moves. I'm considering making the pass bonus depend on how settled the board is, so passing only wins when few useful moves remain.

I'm checking the pass logit against the territory estimate: a margin of +10 with few good moves left makes passing win, while behind by 5 pushes the logit low enough that filling my own eyes (eye penalty -12) still loses to passing.

I'm laying out the layer budget for the score feature: territory indicators and a komi bit go into the first layer, a single score unit is computed in the second, and it gets carried forward to feed the pass logit in the output layer. The pass bias is set so passing only wins once the estimated margin is clearly positive.

