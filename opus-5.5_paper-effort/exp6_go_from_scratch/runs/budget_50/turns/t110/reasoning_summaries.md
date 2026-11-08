I'm looking at the opening move distribution: the first move is almost uniform over the interior, though 13 of 64 games land on the first two lines. The early phase prior penalizes those edge lines, so I want sharper openings, like tengen for black on 9x9, which is a standard strong choice, and I'm leaning toward favoring 3rd-line moves in the early phase.

I'm setting the early-phase edge penalties and the center bonus, so tengen gets the strongest preference on an empty board. The values are tuned so the penalty fades linearly as stones are placed, and I'll test with the E1 base.

