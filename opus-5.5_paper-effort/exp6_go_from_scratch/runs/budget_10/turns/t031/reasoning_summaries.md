The strong bot always opens tengen as black, and its early moves cluster on lines three and four, with very few on the edge. Ours lean heavily toward edge moves, which is a clear imbalance worth fixing. I'm now going to redesign the positional component to get more contrast and a better shape.

I'm planning an edge penalty that fades as stones are placed, using a global unit that decreases linearly until about move 40. I'm also thinking about scaling the positional evaluation with a single temperature factor applied across the components.

Now I'm weighing whether J is too eager to make contact, since adjacent play to our own stones early reads as slow, and diagonal or one-space jumps seem better. I'm considering adding an explicit liberty term, roughly +0.3 per empty neighbor of the candidate point.

I'm working through how to make the positional term more decisive: scaling the edge penalty and the influence kernel so that points near our own stones are discouraged, instead of relying on a soft threshold. I'll check the results with a few more game passes.

I'm settling on the influence penalty: with the stone-adjacency weight raised to about 0.6, a point next to one of our stones nets out to roughly zero, and points beside two or more get pushed negative. Contested points sitting between our stones and theirs read as near-neutral in influence, so they keep the full base value. I'm also thinking the opponent-territory term can carry the same logic, and that the value aggregation kernel should weight the point itself highest, with orthogonal neighbors next.

I'm weighing whether to run a positions pass to check the new landscape before playing more games, since I can't test many variants at once. I'll stick to reasoning analytically and use the positions passes, with the games passes after that.

I'm setting the positional scale to 2.0, with a static prior that penalizes the first line and second line, and a phase unit that fades the edge penalties out by move 40. I'm now drafting the liberty term and the first-line and second-line weights.

I'm now setting the tactical weights: capture and escape values, a per-liberty penalty for self-atari, and an eye bonus, with the escape term netting out slightly negative when the candidate has one empty neighbor.

I'm weighing the tactical side next: the tenuki and pass values need to stay sensible against the positional landscape, which could span roughly -10 to +8 at this scale. Late in the game, deep own territory should sit below the pass value, while contested and dame points stay positive. A remaining concern is the self-atari penalty on edge moves filling dame next to opponent stones, which could push those points to about -9.

Now I'm setting the phase term: the global unit that counts stones reads from the board sum, and I'll give it a negative weight on the first-line points so the penalty fades out by move 40. The pass logit will follow the same count, so early passes stay unlikely.

I'm adding a small bonus to the tengen point in the prior, since the strong bot opens there even though it gets occupied early anyway. I'm also checking how White would likely answer tengen, looking at the open points around it, such as the 3-3 and nearby point positions.

I'm checking whether the tactical bonuses can outweigh the own-territory penalty when a group is in danger, since an escape worth about +8 may lose to a penalty near -10.

I'm tuning the tactical weights so the capture and escape bonuses stay large enough to matter, with the self-atari penalty set so an escape with one liberty nets out slightly negative. Late in the game I expect the own-territory and opponent-territory penalties to push play toward passing, but there's still one endgame question I haven't settled, about how dead opponent stones should be handled.

Next I'm restructuring build.py so the positional scale is applied cleanly, starting with the parameter setup.

