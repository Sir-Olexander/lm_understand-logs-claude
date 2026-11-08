I'm checking the attack term: for opponent stones, each liberty count l adds a score of 2 minus l, so one liberty helps, two is neutral, and three or more start to hurt. That means approaching a lone stone with four liberties gets penalized, which may be acceptable. Now I'm checking the units on the s value.

I'm checking that the Uns formula simplifies correctly: the empty and same-colour neighbours each add one, and the opponent-stone terms subtract, which matches the expected count. Next I'll verify it on a few test positions, covering a single-stone capture, a three-stone capture, and an escape from atari.

I'm writing out a few more test positions: one where black plays inside an eye at the corner, and one where black plays a move that leaves its own stone with a single liberty.

I'm working through test positions for the attack and capture logic: a row of white stones with a single liberty at (5,8), which should register as an atari and get captured, and a variant with liberties at both ends where I expect false alarms at each end. I'm also thinking about checking the empty board to see the prior distribution.

I'm setting up a few test positions, a capture case, a three-stone capture, and an escape from atari, each with a fixed expected move. I'm writing out the board rows for the first of them now.

I'm now setting up a test for an eye at the corner: a black group with a gap at (0,0) that white cannot fill without being captured, and I'm sketching the rows to check that.

I'm laying out a few more test positions: an eye in the corner that should stay almost unplayable, and a self-atari case where black's move at (1,0) would leave its stone with a single liberty.

I'm writing out a few more test boards: a white-to-move capture where white's stone at (4,4) is in atari, and a long row of white stones with liberties only at the two ends, which should flag atari at both.

