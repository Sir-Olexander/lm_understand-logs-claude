I'm checking the n5, n4c, and Uc values, and looking at the false-positive positions. Item 66 is the 7th entry in the pos10 set, which maps to game 7 at ply 98, where the top move shows a pass with probability 1.0, and I need to work out why that index corresponds to that game.

Checking that ply 98 in game 7 is the position behind item 66, I see the top move is a pass at probability 1.0, which seems odd for a board where self-atari was the alternative, though it's late in the game so it may be reasonable.

