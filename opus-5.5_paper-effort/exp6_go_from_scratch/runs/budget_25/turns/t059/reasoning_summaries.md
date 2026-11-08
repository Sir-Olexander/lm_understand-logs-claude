I'm going back over the board state around ply 92, checking whether the stones at (7,7) and (7,8) are really the opponent's invasion of our bottom-right corner before deciding whether to pass.

I'm checking the stones at ply 94 and seeing that the opponent's stone at (8,8) has only one liberty, at (7,8), so it could be captured. Passing seems wrong here given that, and I'm wondering whether the pass logit is set too high in the late game.

I'm checking the capture at (7,8): the opponent's stone at (8,8) has only that one liberty, and nothing around (7,8) blocks playing there. The capture bonus should push the logit well above the pass, so passing here looks like a mistake.

I'm checking the atari at (8,8): its only liberty is (7,8), and (7,8) is empty with no friendly neighbors to block it, so the capture should be legal and worth a substantial bonus, which makes the pass at ply 94 look even stranger.

I'm reconsidering whether the stone at (8,8) really belongs to the opponent, since my color assignment for the net may be backwards. If the net is white, the stones I marked as ours are the invaders, and passing would be reasonable. I'm going to redo the analysis with 'X' always meaning our stones.



