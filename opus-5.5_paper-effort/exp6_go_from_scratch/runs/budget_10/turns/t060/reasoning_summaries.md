I'm comparing the connect point at (4,5), which scores 7.0, against (2,4) at 6.8, with the rest around 6. The connect point's +3 cut-connection bonus is in there, but the position penalty and the base value differ too. The margin is small, so I'm considering raising the cut-connection weight to 5 to make connecting clearly preferred, since the cut is a serious threat. Position 1 showed no such pattern, since it came before white's cut.

Checking the remaining positions now: 13 and 9 look like captures or endgame moves that make sense, and 33 is a capture too. Pass is -5.1 in position 13 against a best of 3.8, so that's clearly not chosen. The opening landscape is fine as it stands. Next I'll plan the remaining passes, with the next games pass testing the current design.

I'm checking the remaining self-atari and ladder cases, and the capture logic should cover the weaker opponents' big blunders. The connection-cut weight is the one thing left to tune, so I'll raise it to 4 and run the games again.

