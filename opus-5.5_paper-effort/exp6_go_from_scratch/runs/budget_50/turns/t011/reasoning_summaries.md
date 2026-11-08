I'm comparing the board states after move 62 and the opponent's reply at 80, which seems to have captured stones at (6,8) and (7,8). I'm checking whether the net's next move at 71 brings the board back to an earlier frame.

I'm checking the board states again and confirming that playing (7,8) recaptures the stone at (8,8), which brings the position back to where it was before. So the cycle of captures is repeating the same position. I'm also noting that we're behind on territory at the top, so our strategy needs to change beyond these local fights.

I'm thinking about the strategy here: the strong bot is pulling ahead on territory, roughly a ten-point margin, so I need to stop passing early and avoid repeating positions. Next I want to look at a typical losing final board to see how the territory is being divided, and check whether our opening moves were sensible.

