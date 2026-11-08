I'm checking the corner position, where we play (0,0) with certainty. Our X sits at (0,1), the opponent's stones at (1,0) and (1,1) are next to it, and I'm working out which liberties that opposing group still has, including (0,0) and (0,2).

I'm checking whether the opposing group's only liberties near the corner are (0,0) and something far away, since (0,2) isn't adjacent to any of their stones. Playing (0,0) would leave our two stones with just one liberty, so I'm weighing whether that self-atari leads into a repeating capture cycle.

I'm trying a different angle: instead of suppressing the self-atari move, I'm looking at the recapture. Once they capture my two stones, their single stone at (0,2) would have only (0,1) as a liberty, so I'm checking what happens when I retake from there.

I'm weighing whether to accept this self-atari repetition risk and look at the broader pattern: eight of the pass-20 losses, about 6% of games, end with positive margins, so a fix there could lift the win rate noticeably. I'm also considering a fix that doesn't depend on the opponent passing, though I can't measure the margin to decide when we're safely ahead.

I'm checking whether the opposing group's capture bonus at the first-line throw-in is really the R5 flag or just the R4 suppression, so I'm going to run the flags and group sizes for those losing games.

