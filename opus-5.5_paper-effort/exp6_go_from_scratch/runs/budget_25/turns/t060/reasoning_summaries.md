I'm checking why the net would pass when a capture at (7,8) looks available. Its neighbors are three O stones, and the lone O at (8,8) hangs on that same point.

I'm working through the capture count at (7,8): the adjacent O stones at (8,8), (7,7), and (6,8) make up the O groups there, and I'm estimating the penalty terms and positional values for that point to see whether the logit lands near 18.5 before comparing it against the pass logit.

I'm checking the capture count at (7,8) and the pass logit, and the numbers suggest capturing should clearly beat passing. Since the network still passed at 1.0, I'm questioning whether the move is legal, the ply parity, or the board state used. I'm verifying that the stone at (8,8) would have zero liberties after the placement, which would make the capture valid.

I realize that the capture at (7,8) is actually a ko: the opponent's stone at (8,8) just captured my stone there, so recapturing immediately would be illegal, which explains why passing was the only sensible choice.

I'm now looking at why the net passes so often even when dame points remain on the board. Those dame points would be worth a point or so each to whoever fills them, and most of the lost points come from passing while they're still open. My guess is the pass logit is too high late in the game, so a dame move never beats it.

I'm reasoning that the late-game pass logit sits too close to the dame-move values, so the network keeps passing even with neutral points left to take. I'm considering a phase-dependent bonus that favors edge moves later in the game, and noting that repetition losses and passes made while our own groups are in atari need a separate look.

I'm going to check the repetition-loss games from pass 8 to see whether they share a pattern.

