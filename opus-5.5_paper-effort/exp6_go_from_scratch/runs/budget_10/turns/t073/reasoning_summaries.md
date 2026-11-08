I'm tallying the game 59 score at the first pass: black has 43, white has 35 plus 7.5 komi, so white is down by 0.5. I'm checking the listed dame points and their neighbors, starting with the one at (6,2), where my white stone at (6,1) looks like it may sit inside black's area and could be dead.

I'm checking the remaining dame points around the white stones at (7,5) and (7,1), and I don't see any gains there. The candidate moves are a pass at 0.974 or (7,7) at 0.026, which seems to confirm the game was already lost by half a point. I'm also looking at the earlier game 17 losses, where the close results look decided in the middle game rather than the endgame.

I'm weighing the final config against the earlier passes: passes 4 and 5 had the best pooled results at 76%, but each has a known flaw. Against the weaker practice bots everything is near 100%, so the evaluation score probably depends on how the config handles the stronger opponents.

I'm comparing the pass results, which were 76, 76, 72, 70, and 69 percent, and the decline isn't explained yet. I'm wondering whether a bug crept in with pass 6, since its changes were the exact F and F3 seeds, sa_kill 1.5, and tengen. Exact F3 produces more false semeai assessments, roughly 12k against 6k, so the engine avoids more single-liberty moves near its own groups. That passivity in boundary fights could be the real cause, and the mix variant lowers false assessments to 9.

I'm weighing whether the false single-liberty assessments are actually the problem: with the stricter weight, moves that are safe but border a large healthy group get penalized, and those are often the exact boundary and hane moves the endgame needs.

I'm weighing whether to soften the penalty for a single-liberty move next to a probably large group, rather than making the seed more generous. Generous seeds would fix the false assessments but bring back the missed true ones, so I'm considering a linear density-based adjustment instead.

I'm settling on a density-based softening: the penalty for a single-liberty move would drop when the adjacent own wall is large, so the true boundary moves stop getting punished. I'll check how many true and false self-atari cases this catches across a few weight values.

