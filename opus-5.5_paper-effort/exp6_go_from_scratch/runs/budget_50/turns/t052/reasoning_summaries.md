I'm checking example 1: the move at row 1, column 8 would leave eight of our stones in self-atari, which the model put at about 7% and sampled anyway, even though the top choice was at 93%. Next I'm looking at example 3, where playing at (3,0) would join the stone at (2,0) and I need to know whether that group still has liberties.

Now I'm checking whether playing at (3,0) would leave the joined group with any liberties, since my engine counts just one left after the connection. The probability is almost certain, so I'm wondering if the move escapes a stone in atari, and the overcounting of adjacent stones from the same string is a problem I'll need to fix.

I'm weighing a max over the four neighbouring contributions instead of a sum, since it avoids the double-counting but might underestimate liberties when two healthy strings join. That underestimate matters, because connecting moves between weak groups are often the important ones.

I'm still working out whether a max over the neighbouring contributions could replace the sum without undercounting liberties, since the sum double-counts stones from the same string. A per-corner correction would need too many units, and a max could underestimate when two healthy strings join, which is exactly where connecting moves matter most.

I'm checking whether a same-string double count can be fixed locally: two 2-liberty stones touching p may belong to one string, and I'd have to tell that apart from two separate strings.

I'm now trying to bound the overcount from corner-adjacent stones, but computing per-corner correction units would need class information that only arrives after the layer where the U2/U3 units sit, so it doesn't fit as things stand.

I'm still stuck on the liberty count: a per-stone sum double-counts a string that touches the move point twice, and I can't cheaply tell those cases apart. A per-corner correction needs more units than I have, and the class information it depends on arrives too late in the layer order. I'm considering an alternative where each 2-liberty string contributes once regardless of how many of its stones touch the point.

I'm weighing whether to count only the best adjacent own string rather than summing every stone, since capping the contribution at the strongest neighbour would avoid the double-count, though it might still undercount when two separate healthy groups meet at the point.

I'm working out a way to count the adjacent own-stone liberties that doesn't double-count a string touching the point twice. One option is to take the first two-liberty neighbour once instead of summing them all, but the circuit layers don't have room for a clean min-style unit at this stage.

I'm weighing whether the self-atari cases come from the double-counting or from something else, so I want to categorize them: for each one, I'd check whether two adjacent own stones sat in the same two-liberty string, and whether the top move's probability suggests the move was deliberate.

