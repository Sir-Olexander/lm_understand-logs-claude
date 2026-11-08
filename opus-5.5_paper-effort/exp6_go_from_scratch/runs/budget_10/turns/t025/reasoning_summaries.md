Barely any improvement, from 568 down to 557. The far failures seem to be groups whose stones are far from seeds, since the e≥2 stones aren't any closer, so either all seed types need to be considered or these are long chains with few liberties.

I'm weighing whether a group with three or more liberties but a liberty stone far from any seed can be handled by tweaking the capture bonus, since a fake capture might otherwise slip through. I also want to check whether the self-atari penalty is actually doing anything, because adjacent false atari neighbors already zero it out, which would explain the game-92 failure.

I'm considering whether a fake capture could still slip through: the self-atari penalty currently gets zeroed out by false atari neighbors, so it isn't doing its job in cases like game 92. One way to address that would be to require some consistency before awarding a capture bonus, though the far-from-seed groups make that tricky to judge.

I'm settling on a layout for the next version: separate planes for stone status, eyes, self-atari, a local shape term combined with the empty-count kernel, and positive and negative influence, which comes to roughly 488 channels.

I'm laying out the output for each move as a sum of neighboring stone status, eye and self-atari penalties, a local shape term, and influence-based terms that push toward filling deep own territory. A pass logit gets its own bias and a term tied to stone counts, and I still need to account for dame points between both colors.

I'm laying out the influence function next: for each empty point, a kernel-weighted sum of nearby stone colors, with weights falling off from 1.0 for orthogonal neighbors down to 0.2 at distance three. Working out the maximum magnitude when a point is fully surrounded, the sum comes to roughly 4 + 2.8 + ... and I'm still tallying the remaining terms.

I'm working out how far the influence values can reach: a point with one own stone adjacent comes to about 1.0, while one at knight distance only gets roughly 0.35. The threshold at 0.5 would mean adjacent points count as territory and distant ones don't, though edge points, which have fewer neighbors, may be easier to claim than the heuristic suggests.

I'm settling the move-value aggregator: each candidate point p sums a distance-weighted term over nearby empty points, rewarding control and penalizing points already held by the opponent. The open question is whether a move actually flips points to my side, since playing at p only raises influence near p, and points the opponent already controls shouldn't count as gains.

I'm setting the move value as a sum over nearby empty points, weighting control gains and penalizing points the opponent holds, with the penalty set at about half the gain. Early on, a center point scores roughly 3.6 against about 2.1 for an edge point, so the value leans toward the middle of the board. I still need to check how deep own territory gets treated late in the game.

I'm settling the move value as a sum over nearby empty points: control gains are rewarded, points the opponent already holds are penalized, and the penalty runs at roughly half the gain, so an early center point scores higher than an edge one. Still checking how deep own territory is handled late in the game, since its points would be strongly penalized and push the value below passing.

Next I'm fixing the influence weights so spacing is handled by the kernel, and checking the orthogonal own-stone term first.

I'm settling the weights for the local shape term: contact with an opposing stone gets the largest bonus, diagonal and two-step shapes get smaller ones, and direct adjacency to my own stones gets none, since the influence term already handles that. Next I need to check whether connecting two of my groups is valued enough, because a move that links them is worth more than the pattern scores show, and the one-ply tactical check doesn't capture that.

I'm settling the static prior with milder values for the first two lines, and I'm setting the pass logit to a bias plus a stone-count term, which comes to about -3.4 late in the game.

I'm checking whether a dame point ends up above passing: its value comes out around +1 to +2, so it stays above the -3.4 pass logit. For an own-territory point late in the game, the fill value is near -1.5 after the diagonal shape terms, which is still above passing, so the bot would keep filling its own area and the open question is whether that wastes moves if the opponent passes.

I'm working out a pass feature based on how many empty points remain that are not true eyes, since filling my own territory could reduce my eye space and let an invasion kill the group. I'm weighing whether to raise the pass logit only in very late stages, when the stone count passes about 70, so the bot doesn't keep filling indefinitely.

Now I'm working out the layout for the new L6/out construction: the POS term at unit 324+x, where each point's influence is a kernel-weighted sum of own minus opponent stones, minus a constant and a penalty on occupied points.

Still working out the layout for the influence features: the negated influence and occupancy penalty go into the NEG units, and the empty-point term splits into a constant bias plus negative stone weights, with the offset chosen so the pre-activations stay non-negative.

I'm writing the code for the influence and occupancy features on the positions, then I'll test it on a set of opening and mid-game positions before running full games.

Considering how repetition avoidance could work here: the recapture in that game followed a self-atari I played earlier, so the real fix is avoiding the self-atari itself rather than tracking history for PSK.

Now I'm writing the new builder version, fixing the 2-stone-in-atari issue at the root rather than tracking repetition history.

