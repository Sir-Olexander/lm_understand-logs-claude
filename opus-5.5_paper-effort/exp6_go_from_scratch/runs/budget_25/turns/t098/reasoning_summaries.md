The same self-atari pattern keeps recurring: we play a move with near-certain probability, they capture two or three stones, and we only win one back. Since the policy is so confident in those moves, I suspect a large bonus is driving them, likely the capture feature on a big group.

I'm weighing whether the capture bonus on the large group is what's pushing the self-atari moves, and I want to test that by looking at the semantic eval under the v11 parameters. Next I'll pull up the initial moves of games 16, 51, 60, and 71.

