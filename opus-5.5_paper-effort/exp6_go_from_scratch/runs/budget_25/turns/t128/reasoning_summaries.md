I'm at 70% against black's 64%, which is worse, and the Tengen first move didn't help and may have hurt. The noise is about ±4%, so I'm reverting LV[5] to 3.0, back to v16. Repetitions are still at 13 and costing roughly 5-8%. With 4 passes and 14 runs left, I'll try one cheap targeted change to cut the S2R1 repetitions before finalizing, and the idea is that S2R1's last move is our recapture of a single

I'm now checking whether the S2R1 recapture can be told apart from an ordinary single-stone capture: the opponent's stone x sits in atari at s, and s has an empty neighbor p that was also just vacated.

I'm weighing a recapture-specific penalty against a more general pocket penalty, since the pattern probably shows up in ordinary fights too and could cost real stones at the edge.

I'm weighing the trade-off: about 119 small captures in that configuration against roughly 30 repetitions, where each repetition effectively costs a whole game. Missing a small capture usually costs a few points at most, so suppressing the pattern looks worthwhile, but I still need a way to implement an exception for the configuration where there are no empty neighbors and no adjacent white stones.

I'm trying to see whether a single-unit penalty can separate the S2R1 recapture from real captures, and the answer looks like no. So I'm considering whether the eye unit or P2 could be repurposed to detect that configuration instead.

I'm weighing whether to split the spare L6 units to cover the most common S2R1 first-move points, like the corners and their neighbors, and give those points a separate penalty instead of the Uc term. The worry is that the positional nonlinearity that helped earlier might be lost, so I'd rather add an extra penalty unit for those points than replace Uc.

I'm sketching a penalty unit Ux that keeps the single-stone k=1 case but drops the suppression, with a linear form in k, n_e, and the sum of W, so it leaves the eye-filling and one-empty-neighbor captures alone.

I'm working out the constraints on the penalty unit: it should fire only for a single-stone capture with no empty neighbors and no own or opponent stones adjacent, and stay quiet in the eye-filling and one-empty-neighbor cases. The open question is the case where capturing an adjacent atari stone is what saves us, since that would be a true capture and shouldn't be penalized.

I'm testing a unit with weights chosen so it fires on the single-stone, no-empty-neighbor case but stays quiet when there's an empty neighbor or a friendly stone adjacent. The check is that it still penalizes a two-stone group with no liberties beyond the captured point, and I'm noting that it misfires when the adjacent atari stone is a real escape by capture.

I'm working out whether a penalty weight of 60 on the Ux unit would suppress the S2R1 first move, since 30 would leave it at +5 and still get played. The catch is that the same weight also blocks the real captures in this configuration, so I'm weighing whether losing those small captures is worth avoiding repetitions that each cost a whole game, which is 5-8% of games.

I'm listing where the S2R1 first moves landed in passes 19 to 21, and they cluster on corners and edge points a few steps away from them. That gives a set of 22 candidate points, and I'm planning to give those points their own penalty unit at L6 rather than changing the existing Uc term.

I'm estimating how often the captures in this configuration would be lost at the 22 candidate points, roughly 0.25 per game, against about 0.08 per game of repetitions, which each cost a whole game. A missed small capture usually costs only a few points, so the trade looks favorable, though a penalty of 30 would still leave the true captures and the repetitions tangled together.

I'm going to implement the Ux penalty unit with weights α=1, β=1, γ=1, δ=0.5 and a weight of 60, since its pre-activation is k minus n_e minus the summed W minus 0.5. I'm now working through how that pre-activation expands in terms of the O5' values.

I'm checking whether the combined penalty unit can separate the true captures from the S2R1 recaptures: with these weights, the pre-activation reduces to the summed liberty-minus-neighbor count minus 0.5, so the single-stone no-empty-neighbor case lands at 0.5 while eye-filling and the one-empty-neighbor cases stay negative.

I'm checking whether a penalty unit with these weights would also catch the case where our stone next to p is itself in atari and capturing is the only escape. It would, which means a real capture-to-save at these corner and edge points would get suppressed along with the repetitions.

I'm working through whether the atari case should be penalized too, since a true capture that saves our stone would also trip this unit. I'm considering splitting the weights so the two-liberty case and the atari case are handled differently, but I'm not settled on it yet.

I'm listing the S2R1 cycle locations and the first-move points they start from, mostly corners and nearby edge spots, and that gives the candidate set I'm building the penalty unit around.

Now I'm settling on 22 candidate points: the four corners, the eight points next to them, and ten more along the edges. I'm going to set those 22 points aside and revert to the v16 version with LV[5] back at 3.0, so I can build the penalty around them.

