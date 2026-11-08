Looking at the lost games, the strong bot as White is building large territory while our Black stones stay scattered and slow. By the end White holds the top and left, and we're left with the bottom right, well short of the 7.5 komi we need.

I'm weighing an influence-based way to score candidate moves, since our stones keep getting walled in by contact play and territory is the real gap. The layer budgets look workable for a nonlinear influence term, so I'm checking whether the board information it needs is already available at the right depth.

I'm checking whether b can carry a small influence signal through the layers without breaking the gating, which holds at stones but gets murky at empty points where neighboring own units are involved.

I'm checking whether the bounded-unit arithmetic still holds when the influence signal leaks into empty points, since small leftover values there could shift the activations I'm relying on. With a moderate pre-activation margin, the empty-point noise stays below the threshold, so the approach looks workable.

I'm checking whether early-layer positional features can be folded into b's accumulator at empty points, as long as they stay nonnegative and zero at stones. The perturbation budget works out if ε is small enough to keep the empty-point noise under about 0.01, and float32 precision still leaves the reading error negligible.

I'm thinking about which positional features would matter most here, since the strong bot wins on territory and our stones end up clustered. Freeing up budget is the constraint, and I'm checking whether every own color channel is really needed at each point. The main candidates are avoiding wasted moves inside our own solid area and avoiding empty triangles from stones packed next to each other.

I'm defining a territory estimate for candidate moves: each empty point gets an influence score for both colors, and a move's value is the weighted sum of how much its added influence would shift the contested points nearby.

I'm working out a cheap bump function for each point's influence difference: a three-relu triangle gives max(0, a − |I|), which peaks where the influence is contested. That's still too many units if applied everywhere, so I'm looking for a lighter form.

I'm considering a cheaper bump function: three relus per point give a triangle that peaks where influence is contested, but applying it everywhere is too costly, so I'm checking whether sampling a sparse lattice of points would still capture the territory signal. Then I'd weight each sampled point by its distance to the candidate move. I'm also weighing a simpler frontier test, flagging moves near both own and opponent stones, though that needs a nonlinear product-like term.

I'm weighing whether a simple contact test would do the job: flagging points where own and opponent influence both show up, using a min of the two presence counts. That needs only one ReLU per point, so it fits the budget, though I'm not sure heuristics like this will beat what an RL policy has learned.

I'm reviewing the game records to see where the losses come from, and it looks like the strong bot mainly wins by building territory. I'm wondering whether our bot's stone losses are mostly from tactical failures or from positional ones, since the margins suggest White simply controls more of the board.

I'm turning to the opening, wondering whether a stronger positional prior would help, since our stones keep landing at the same distance from the opponent's line. I'm considering checking how the RL bot handles contact moves early on, and whether its opening choices in past games could offer a useful pattern, without fitting weights to the data.

I'm sketching a territory-based move value: each point near a candidate gets weighted by how contested its influence is, so moves near the frontier score higher than moves inside settled areas.

I'm settling on a three-unit triangle bump, since that's the only form I've found that cleanly gives max(0, a − |I|). Applying it at all 81 points would cost 243 units, which is too much, so I'm weighing whether a cheaper two-unit approximation could work instead.

I'm settling on a leaner setup: rather than a three-unit bump at every point, I'm weighing whether to compute it only on half the points or to restructure the budget, since the tactical machinery already takes most of the network.

I'm now working out how the bump units can read the influence values: since the influence only depends on O minus P, it's a linear function of the previous layer's activations, so the bump can be computed at any layer from L2 to L6. The open question is how to feed those bump outputs back into the value, which is a linear combination and has to be accumulated into the next layer's accumulator.

I'm now weighing a coarser set of sample points for the territory bump, roughly 25 lattice points, which would cost around 75 units and fit the budget at the output layers. I'm also noting that the bump outputs have to be linear at the last layer, so I can't gate per point without extra units.

I'm comparing a territory-bump feature placed at layer 5 against the leftover unit budget, roughly 22 sample points there and a few more at layer 6, to see whether a sparse set of positional readings can fit alongside the tactical machinery.

I'm weighing a simpler contact term that uses the minimum of own and opponent influence at each point, which costs about one unit per point. It would reward contested boundary moves, but I doubt it captures territory gain well. I'm also considering a linear penalty on moves surrounded by many of our own stones, to counter the overconcentration that the strong bot seems to exploit.

I'm thinking that a simple positional prior could work: spread stones at distance two from our own groups and approach the opponent at distance two rather than making contact, while keeping solid direct connections only where tactics demand it. Before settling on any heuristic, I want to look at how our bot actually lost a game, move by move with its top five candidates, to see where the decisions go wrong.

