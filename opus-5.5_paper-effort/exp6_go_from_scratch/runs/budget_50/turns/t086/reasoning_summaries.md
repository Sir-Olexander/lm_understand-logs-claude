The large move logits are coming from the legal mask, since illegal occupied points still feed into the max over the first 81 entries, so those values can be ignored. The pass probability sits near 1.0 with its logit around -2.2, which suggests the legal moves all score far below it and these are end positions where passing is the right call anyway. I also need to account for the estimate running high by about 7 to 10 points.

I'm checking the perspective used for the true margin: when we're white, it should be white minus black plus 7.5, not black minus white. In the close-loss positions, passing was not the real mistake, since the game was already decided by earlier moves.

I'm wondering whether the estimate runs high in black's favor, using pos 0 as the example: est 4.5 against a true -2.5. The territory proxy weights each point by its own-neighbor count times 0.6, which may explain part of the gap. Since the losses were already decided earlier, I want to look at the final boards of the close games for dead opponent stones sitting in my territory, because those cost points on both counts.

I'm now asking whether dead opponent stones left inside my territory at game end are costing points, since under literal scoring they count for white and make nearby empty points neutral. That would matter most in the close games, so I want to check whether any opponent stones in my area could have been captured before passing.

I'm checking whether opponent stones with only a couple of liberties are sitting inside my own territory, since those would count against me at the end. I'll scan the board at my first pass and see which ones are fully surrounded by my stones.

