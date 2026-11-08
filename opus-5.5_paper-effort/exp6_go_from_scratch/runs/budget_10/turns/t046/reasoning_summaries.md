The new F3 rule is clearly more conservative: for two-library groups it correctly keeps T at 1 about 98% of the time, versus 83% before. The trouble is the groups with three or more libraries, where T gets set falsely because no F3 is found nearby.

The new rule is worse on the two-liberty groups, so the edge-based approach gives only a marginal improvement. The false-atari problem doesn't look solvable just by tuning reach, so I'm considering a different angle: reducing the damage from false captures. The concern is that a false capture can trigger self-atari avoidance, and I'm weighing whether to accept missing some true captures in exchange for fewer false ones.

I'm considering a middle ground: lowering the self-atari kill weight so it scales with how dense the neighboring white stones are. An isolated stone would still allow a true capture, while a dense area would mostly block it, which should cut false-atari damage without giving up too many real captures.

I'm checking whether a dense group of white stones would be missed by the capture check, which I estimate at about 7% of true atari stones. I'm also looking at game 11's move at (0,8): its neighbors are one white and one black stone, and the adjacent black stone at (1,8) has two liberties, so the self-atari count depends on the kill term.

I'm going to look at the largest blunders for the current version and sort them by category, starting with the first dozen and their surrounding context.

