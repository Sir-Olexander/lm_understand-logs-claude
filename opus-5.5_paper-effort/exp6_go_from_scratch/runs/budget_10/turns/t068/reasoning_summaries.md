I'm finding that 33 of the 48 blunders are false captures, where the opponent group looked like it was in atari but actually had two or more liberties, so our stone got captured instead. These self-atari moves bypass the suicide rule through a kill, and they lead to repetition losses. Next I want to count the true captures that would be lost if I don't allow the suicide-kill exception, and weigh that trade-off.

I'm now checking the moves that were flagged as likely captures but had at most one liberty and no third neighbor, comparing how often a capture actually happened against the false alarms. I'll count across all such moves, not only the blunders, and also measure how big the true captures were.

