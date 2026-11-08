I'm checking the board state at ply 96, confirming the black stone at (7,1) matches net 64. Then I'm tracing whether our sequence of corner moves and captures would bring the position back to an earlier one, comparing the board after ply 100 against ply 96 and noting that the stone at 64 differs.

I'm now considering how the game could end in a repetition loss, where the opponent passes and we keep replaying a stone that gets captured, so the board cycles back to an earlier position. Since repetition costs us the game even when ahead, I need to find a way to break that cycle, though without the move history I can't directly tell which earlier boards have occurred.

I'm working out how to avoid repetition losses: the pass-and-recapture cycle recreates an earlier board, and without move history the network can't see that. The practical fix is to discourage pointless self-atari invasions late in the game, so passing becomes the better choice, and I'm weighing whether the self-atari penalty should sit below the pass logit.

I'm settling on a self-atari penalty of -20, below the pass logit, so the network passes instead of throwing single stones into settled territory. Captures will still override that penalty, and I'm counting on repetition-prone invasions mostly disappearing late in the game.

I'm wondering about the margin, which averages -11.5, and whether the real issue is weaker positional play against the strong bot. I want to look at a typical game's opening and print a strong game at plies 10, 20, 30, and 40 to see where it goes wrong.

