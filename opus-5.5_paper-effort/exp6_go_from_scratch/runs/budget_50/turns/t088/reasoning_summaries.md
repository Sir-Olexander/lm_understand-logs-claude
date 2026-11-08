I'm checking the early-game stats: we open on the edge far more often than the opponent does (16% vs 6%), and we lean toward isolated moves while they favor contact. My sense is that I should cut back on early edge moves and push for more contact, since my prior already penalizes the first line. Some of those edge moves may still be tactical, like captures or escapes, so I want to separate those out.

I'm weighing whether to push the line-0 and line-1 penalties harder and shift the contact weights, but edge moves still matter in the endgame for blocking and dame. So I'm considering making the edge penalty depend on game phase, measured by stones on the board, so it fades as the game goes on.

I'm settling on the phase-dependent edge penalty: strong early (line 0 at -3, line 1 at -1.2) and relaxing toward -0.5 and 0 late in the game, using a stone-count feature clamped to a 0-to-50 range. I also want to try slightly stronger contact weights in the next test run.

