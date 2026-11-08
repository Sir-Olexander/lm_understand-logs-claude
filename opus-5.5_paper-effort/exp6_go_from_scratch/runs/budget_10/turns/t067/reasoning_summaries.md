The repetition losses are the biggest leak: five games, four of which we were winning, works out to roughly 5% of games. Blunders also look high, probably from the binary A false captures. Black is losing most of the close games at komi 7.5, and I only have one pass left.

I'm weighing whether to spend my last pass on a final check, since any changes made after it wouldn't get tested. Before deciding, I want to pin down the repetition losses: the pattern is that we self-atari two or more stones, the opponent captures with a stone that is itself in atari, and our recapture of one stone recreates an earlier board position. I'm looking at whether that recapture can be detected from the board before we play it.

I'm working out when a recapture reproduces an earlier board position: if we recapture at the point where our original stone sat, and the other captured point is the one we just added, the board returns to its prior state. Since their stone has only one liberty, that recapture point is forced, so the question is whether the captured pair splits so that the recapture repeats position.

I'm checking whether the blunders and repetitions in pass 9 share a common cause, since the self-atari moves seem to come from the false captures and missed self-atari checks rather than from the recapture logic itself.

