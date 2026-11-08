I'm checking the self-atari cases, and most of the high-probability ones look deliberate. For the case where the move connects to a two-liberty string of its own, the count depends on how many stones it touches: two would leave it safe, one would leave one liberty.

I'm checking whether the connecting moves that come out as wrong still get chosen, so I'm looking at the cases where they fill a point next to opponent stones or an eye. Next I want to fix the penalty for stones left with only one liberty so that the escape bonus only counts when the result is actually safe.

I'm working out how to make the escape bonus fire only when the result is actually safe. The idea is two units whose difference equals the neighbor-based term whenever there are at least two liberties, and zero otherwise. That needs about 162 extra units in layer 6, which fits within the 350 I have spare. Then the capture bonus doesn't need that safety check at all.

I'm checking the cases where the move connects to our own string with no empty neighbors, and for the failed escapes, where only one empty neighbor is left, to confirm they really leave a single liberty and are correctly penalized. Next I want to print a few boards from the own_libs2_e0 and noown_e1 categories with the move highlighted to see why those moves get chosen.

