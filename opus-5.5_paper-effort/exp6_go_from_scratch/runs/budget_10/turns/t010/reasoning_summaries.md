I'm checking the atari flag against actual liberty counts: when there is exactly one liberty it fires every time, but at two liberties it's wrong about 15% of the time, and at three or more about 4%.

Now I'm checking the terms: the second flag shows about 84% hits at two liberties and only 7% at three or more, while the big groups make propagation depth a worry.

I'm now working out how to fold richer seeds into the propagation: the first pass gives the pair signal H and the simple seeds F1 and F3_1, and the second pass adds neighbor seeds, which means masking so that stones with no seed stay at zero.

I'm weighing whether richer seeds can be added in the second pass: if a stone has an adjacent neighbor with simple seed strength, that can be pulled in by masking so unseeded stones stay at zero, and the constant K is set large enough that non-own stones never leak through. That extends reach for the e≥2 seeds, but it doesn't yet carry the pair-seed information, so I'm wondering whether the pair terms could also be computed in the first pass, since the edge units would need roughly 288 slots and the first pass only has about 188.

I'm weighing whether the pair-seed condition can go into the first pass as well: for each edge, a seed that fires when both stones are own and their exponents sum to at least two, plus per-point single seeds for isolated stones. That would remove the need for the separate H pass, but the slot count is tight.

The pair-seed approach costs about 612 slots against a 512 budget, so putting every edge in the first pass won't fit. Next I want to know whether the false positives come from propagation depth, meaning the stone sits more than three steps from any seed, or from seed incompleteness, meaning no seed fires near it at all. Computing each stone's chain distance to the nearest seed will tell those apart.

