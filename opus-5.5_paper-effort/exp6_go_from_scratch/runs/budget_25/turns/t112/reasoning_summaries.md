The strong-only numbers show v12 with more self-atari stones and more stones lost than v9, so the V-veto didn't cut self-ataris after all. Maybe the lower uc reparam made some other self-ataris easier. I think the tactical side has run into diminishing returns with the current architecture.

I'm now weighing whether a territory-style positional evaluator could beat the current contested-ness and center-preference heuristics, since the non-wipeout games are where we lose. I'm working through an influence formulation where a move's value comes from the weighted count of undecided points nearby, but computing that per point would cost more than the budget allows.

I'm now weighing a stronger proxy for contested-ness: right now it only looks at the point itself, and it scores empty regions as zero. I'm considering whether raising the weight on empty, undecided areas and lowering the contact weight would push play toward big points, though the edge-penalizing line values might pull the other way.

I'm now weighing a color-specific tweak, since as black we only win about 55-72% versus 78-87% as white and komi means black needs more territory. Making the positional term depend on color looks costly, because the current formula is linear and a color-dependent product wouldn't fit the budget without duplicating units. I'm wondering if the output layer could read color directly instead.

I'm weighing a territorial experiment next: raising the empty-area weight to 1.5 and dropping the contact weight to 3.5, with the line values flattened so 3rd and 4th lines beat the center. I'm not sure the tradeoff is sound with eight passes left, but each run is noisy enough that only a large effect would show up anyway.

I'm going with the territorial experiment now: raising the empty-area weight to 1.5, lowering contact to 3.5, and flattening the line values so the 3rd and 4th lines outrank the center.

