I'm noticing the early top-5 values are uniform at 0.0323 because the line-3 and line-4 points have equal logits, which seems fine. Several games ended by repetition, some we lost after making the repeating move, one we won when the opponent repeated, so I need to check who made the repeating moves.

I'm checking whether games end on two passes around ply 90-100, which would mean we passed even though our pass logit is -15. I want to see the final positions and our last few moves' top-5 values to figure out when we'd actually pass.

