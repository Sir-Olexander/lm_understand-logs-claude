I'm checking the repetition pattern here: a self-atari of two stones, the opponent captures, and we recapture one, which ends in a repetition we'd lose. I also notice the opponent falls into the same shape in several games where we ended up winning.

I'm working out how the self-atari leads to the cycle: we lose two stones, retake one, and the board returns to where it started. I'm checking whether the suppression should depend on whether the move touches our own stones, since that's where true captures could be wrongly blocked.

I'm weighing a softer version of the suppression: it would only be cut back when an intersection-based check says the move sits inside an eye, and I'm considering a separate rescue for the case where the move has no own neighbors and no empty neighbors. The concern is that a blanket suppression of 2.5 would also erase the penalty on the self-atari cases I want to keep, so I'm looking at making it conditional.

I'm exploring whether a rescue term could work instead, since the (0,0) case, where the move has no own or empty neighbors, is always a capture and currently gets a value of 4. The issue is that scaling down that unit would also weaken the signal for the other cases, so I'm looking at adding a separate bonus for surrounded-by-opponent moves at the output.

I'm weighing a rescue for the all-opponent-neighbor case, since that move is always a capture, by adding a linear bonus tied to the number of opponent neighbors.

I'm weighing whether to drop the blanket suppression in favor of a rescue for the big-capture case, since the veto hurt captures inside eyes. Against that, the no-veto versions score about 75-77% while the vetoed one drops to 65%. I'm now thinking about targeting the "sending two returning one" cycle directly, since it needs our 2-stone self-atari next to a single stone with two liberties and the large opponent group.

I'm pulling up game 44 next, where we play the corner at (0,8), the opponent captures two, and we recapture one. I want to run it through the semantic evaluator to see whether a capture bonus or something else made that move look attractive.

