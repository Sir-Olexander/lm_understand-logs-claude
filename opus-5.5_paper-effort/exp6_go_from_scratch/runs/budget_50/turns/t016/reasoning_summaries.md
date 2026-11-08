The opening so far is weak, with stones scattered on edges and corners, which looks like the result of random sampling. The prior is almost flat across the interior, so the moves come out nearly uniform over the 49 interior points. The opponent, meanwhile, is building a solid connected wall through the center on row 3. I need a much sharper positional policy, and I'm starting to sketch some ideas for it.

I'm now sketching a policy that favors moves touching both our stones and the opponent's, since those contact points are where good play tends to happen. I'm also weighing cutting points and connecting points, and thinking about whether the center opening would be a better first move.

I'm weighing a hand-built positional evaluation, since I can't fit weights to the opponent's moves. The plan is to value each empty point by how much it changes estimated territory, giving more weight to contested areas than to ones already clearly owned.

I'm working out a two-stage computation: first estimating ownership influence at each point, then measuring how contested each point is, and scoring candidate moves by the contestedness they touch, with extra weight on opponent-leaning areas.

I'm trying to make the contestedness term work with a small network: the bump function relu(θ − |I|) has to come from relus on I and −I, and a single extra layer should be enough for it.

I'm checking whether a triangle bump can be built from three relus of the influence value, since that would let each point's contestedness be computed in a single layer. Summing those bumps into the accumulator would still need room in a later layer, so I'm weighing whether to split them across layers 4 through 6 to keep the unit counts within budget.

I'm checking the layer budget: layers 4 and 5 each have about 107 spare units and layer 6 about 188, which is enough for the 243 bump units. I also need the stones masked out of the contestedness term by shifting each empty-point input down by a large constant, so that occupied points contribute zero.

I'm checking whether this evaluation makes sense on an empty board: with no influence anywhere, the value would simply favor interior points over edges. Once a stone is placed, the points near it count as less contested and the far areas as more contested, so the next move would lean toward distant regions. I'm not yet sure that "play where it's contested" is good in general, since safety of weak groups matters too.

I'm now considering whether the move value can be measured as the change in estimated territory, comparing ownership before and after placing a stone at each point. The estimate would use a saturating function on the influence, so the gain at each nearby point is roughly the local kernel weight times how sensitive that point's ownership is to influence.

I'm weighing a distance-weighted influence kernel where nearby stones shape each point's ownership estimate. I'm also considering whether to lower the pass logit so the bot doesn't pass while behind, and to test the influence and escape changes against a stronger opponent.

I'm settling on a pass logit around -2 so the bot stops passing while behind, and making moves inside its own territory costly, especially eye-filling, which gets a large penalty. When only its own eyes remain as legal moves, passing is the better choice.

I'm checking whether capturing a stone could leave my own stone in atari, which would invite a recapture and set up a repeating cycle that I could end up losing under superko.

I'm weighing whether to stop throwing stones into dead areas inside the opponent's territory, since those exchanges seem to drive the repeated positions. Passing instead might cut the repetition risk, though if I'm behind it would still lose the game.

I'm laying out the layer budget for the new network: the spare units in each layer, and where the influence bumps can sit so that the escape gating still fits in layer 6.

I'm splitting the 243 bump units across layers 4, 5, and 6, with 81 in each, so that the three terms for each point land where the layer budget has room. The plan is to sum those bumps into the accumulator to get the value for each candidate point.

I'm now setting up the distance-weighted influence kernel, with weights of 1, 0.5, and 0.25 for neighbors out to three steps. The bump terms feed the value of each candidate point with the pattern +1, -2, +1, and the stone-occupied points get masked out of the contestedness term.

I'm checking the influence kernel's scale: with weights of 1, 0.5, and 0.25, the center point scores about 3.0 on an empty board against roughly 2.25 for an edge point, so the gap is small and the prior will still matter. I'm weighing whether a scale of 0.3 gives a sharp enough opening preference, and I'm also wondering about the bump threshold and whether it should be set at 2.

I'm weighing whether the move values will be sharp enough, since with temperature-1 sampling over about fifty candidates I'd need gaps of roughly 3 to 5 logits between the best few and the rest. I'm also considering a few explicit opening priors, like a strong response to an empty-board tengen, though I'm not sure they're worth the effort. Against this opponent, which has built a connected wall across row 3, I think the most useful change is one that makes the evaluation respond to how the wall splits the board.

I'm thinking about adding heuristic shape features on top of the influence approach: bonuses for moves that extend from my own stones or make contact with the opponent's, a penalty for crowding around my existing stones, and a moderate value for distant points early on. The plan is to add these and then check the resulting logit maps on some sample positions.

I'm listing the concrete changes next: gating the escape term in the output, raising the penalty on unsafe moves, adding the frontier influence values, setting the pass logit to -2, and possibly boosting the tengen prior at the start.

I'm laying out the bump units across layers 4, 5, and 6, each tied to the influence of nearby points and masked for occupied intersections. I'm also writing a helper, add_bump, so these additions can be appended after the base network is built.

I'm reworking the base network so the escape term is gated and the unsafe-move penalty goes up, replacing the old escape term inside the accumulator with gated units in layer 6.

