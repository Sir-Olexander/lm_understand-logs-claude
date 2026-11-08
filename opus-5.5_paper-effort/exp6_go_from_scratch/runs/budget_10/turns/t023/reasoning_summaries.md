I'm checking the final score for game 36: black has 42 and white has 42.5 with komi, so black loses by half a point. I'm also noting a black stone at the top left that sits inside white's area, which still counts for black as one point.

I'm looking at a few points on the board for game 29 where white still has eyes and where we passed at ply 96, even though the opponent kept playing.

I'm working out why passing at move 96 was the wrong call here, even with a pass probability of 1.0. Passing gives the opponent free moves, so I should only pass when they have just passed too or when nothing useful is left. My valuation of boundary points looks too weak, and that gap is what made the pass look better than it was.

I'm going to reconstruct the position at ply 96 by looking at the board after ply 95 and checking which legal moves were still available.

