I'm checking the rendering: the opponent's stone maps to index -1, which lands on the '*' at the end of the string, so opponent stones and the move both show as '*'. The move position gets printed separately, so that's manageable. Next I'm looking at the first example, where the move at (1,2) sits beside our stone at (1,1) and is surrounded by the opponent.

I'm working through the board positions near the move at (1,2), where our stone at (1,1) sits inside the opponent's territory. These throw-in moves seem harmless under area scoring, but they waste tempo while real moves remain, so I'm weighing why they might still beat the alternatives.

I'm pinning down which example to trace: the move at (0,0) looks like a capture bonus from a false atari flag on the big white group, so the late-game throw-ins may just be a scoring artifact. The bigger concern is that we keep losing territory, so I'll print the boards at moves 10, 20, 30, and 40 of a lost game as black to see where the margin starts slipping.

