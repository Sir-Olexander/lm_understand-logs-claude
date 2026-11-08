I'm comparing the two versions for SA prediction on the e≤1 candidate moves. The current one misses about 2,411 true SA cases while the new one misses only 388, but the new one flags more false SA cases, which blocks moves that should be allowed. Since missed SA cases are the costlier error, I'm leaning toward the new version, probably because the remaining misses are mostly false captures.

I'm weighing the trade-off in the new SA version: it misses far fewer true SA cases, but its false positives mostly block moves like filling liberties that don't matter, so I'm leaning toward taking it, possibly softening the SA=1 versus SA=2 distinction. The T_opp false flags on about 8% of groups cause a minor misapplied atari bonus, and the shared threshold T can't be tuned separately for SA, but that's secondary. Since the remaining losses look like self-atari from false captures and SA misses, I'll check the pass-5 losses next, where margins are mostly close and some games lost 43 to 57 stones.

I'm going to reduce the kill weight to 1.5 times the opponent's adjacency, since that keeps isolated groups penalized while letting denser ones through. I'm writing the new seed formulas for the two liability terms now.

Still working through the gating checks on the new seed formulas, and I'm reducing the SA kill weight to 1.5.

