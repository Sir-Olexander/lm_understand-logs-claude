I'm reconstructing a bottom-left sequence on the board: the opponent's O at (8,0) and my X at (8,1), with an O at (8,3) and a capture pending at (8,1) before I play (8,2).

I'm checking whether capturing the O at (8,3) with my stone at (8,2) recreates an earlier board position, since the lone capture here has no friendly neighbors and seems to be the source of the repetition. I'm weighing whether to suppress these isolated captures, but the white stones sitting inside my territory near (8,0) to (8,3) would count against me at scoring if I leave them, so I have to balance the two.

I'm weighing whether a capture can be made safer by playing elsewhere, but the captured O at (8,3) has only one liberty, (8,2), so there is just one way to take it. After that capture my stone at (8,2) gains liberties at (8,1) and (8,3), and I expect the random opponent to keep playing into that area, so each capture might recreate an earlier position.

I'm working out how a ko-style repetition could arise here: if the opponent recaptures at q and I recapture back at p, the board returns to an earlier position, which is exactly the classic ko cycle that a simple ko rule would forbid.

I'm now considering the ko logic: if the opponent captures at q and I immediately retake at p, that is the banned simple-ko recapture, so any repetition must come from a longer cycle with intervening moves. That makes me think about suppressing isolated single-stone captures, but only in some phases of the game, and I'm weighing what that would cost.

I'm weighing the cost of letting these isolated captures stand against the repetition losses they cause, since the random bot's repetition rate is small but not zero.

I'm now checking whether a "pass when ahead" policy would be safer than these isolated captures, since the earlier positions in these random games were ones where we were already well ahead.

I'm settling on a pass bonus that grows once our estimated lead is large, so that when far ahead we pass instead of making the isolated capture. I'm sizing the bonus so that at an estimated lead of 25 it outweighs the capture's logit.

I'm now weighing a pass bonus that rises with our estimated lead but is capped, so that a large lead doesn't make us ignore threats like atari on our groups, which would still need escape moves to compete with passing.

I'm settling on a capped pass bonus, so the pass logit tops out near a fixed value as the estimated lead grows, and a large lead can't push out the escape moves I'd need against atari.

I'm settling the pass bonus numbers: about 8 at a lead of 25, rising to a fixed ceiling so that real captures and escapes still win. That needs one unit for the score and two relu units to clamp it, which fits in layer 6.

