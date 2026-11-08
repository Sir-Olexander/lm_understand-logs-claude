The frontier values look too flat, favoring empty space far from stones over contact and blocking. In position 3, the key moves seem to be around (4,3) to (4,5) and (2,4), with most values sitting between 2 and 3.5. I think I need a fundamentally better positional evaluation, so I'm considering which Go patterns I could encode through local 3x3 features and simulation.

I'm weighing which local Go knowledge to encode: contact responses like hane and extensions, and atari-style pressure that cuts opponent liberties to two or fewer. I'm also considering a feature for my own weak strings with two liberties, though I'd need more spare capacity in the layers to fit it.

I'm weighing whether the policy is too flat: with dozens of mediocre moves sitting in a similar logit range, the sampling keeps picking edge moves like (7,0) and (0,3). Sharpening the distribution with better ordering and a larger logit scale looks like the most direct fix.

I'm thinking about how to make the move distribution sharper, so the policy stops favoring edge moves that get sampled out of a flat spread. The next part of my reasoning is about a simple heuristic approach, opening and contact responses, and how hard it is to hand-design a strong 9x9 evaluator compared with the current net's performance against the weaker opponents.

I'm weighing an influence-based territory estimate, where a move's value comes from how much it shifts the ownership of nearby points. The trouble is that on a sparse board most points look contested, so the gain comes out roughly even across the interior and fails to capture things like blocking next to an opponent stone on the third line.

I'm thinking about a distance-based ownership estimate instead of influence: each point goes to whichever color's nearest stone is closer, so a new move's value is how many points it would claim for us.

I'm working out how to count the points each side would claim, using nearest-stone distance fields that a few ReLU units can approximate. The gain from a candidate move would then be the number of points it would take ownership of from the opponent.

I'm now working out a concrete counting rule: a candidate move gains a point when no opposing stone is at least as close to that point, so the move would claim ownership of it. The gain would be a sum over nearby points, with the claimed amount depending on the distance to the candidate.

I'm checking whether a point at distance k from m is claimed by a candidate stone, which holds exactly when the disk of radius k around m has no stones in it. I think that test can be a single ReLU unit, so each distance k gives its own indicator for whether empty space is claimed.

I'm working out how to count the territory a move would claim: a point m is taken by a candidate at distance k only if no stone of either color sits within that disk, which a single threshold unit could check. That gives about 81 units per distance, and I'm wondering whether partial credit for flipping a point from the opponent is worth adding on top.

I'm working through the gain for a candidate stone at distance k from a point m. The gain is zero when the opponent's nearest stone is already closer than k, and it becomes +1 when the opponent's nearest stone is exactly at distance k, since the point becomes tied.

I'm working out the gain for a candidate at distance k from a point m: it's zero when the opponent's stones are already closer, and the case where the nearest opposing stone is exactly at distance k gives +1 when the point becomes tied.

I'm refining the gain formula so that it doesn't overcount: a point already in my own area shouldn't reward a move next to my own stones. The correction term depends on whether that point is currently closer to my stones or to the opponent's.

I'm working out the gain for a candidate stone. Each term is a product of two "empty disk" conditions, which I can only write as a single ReLU if I compute the inputs one layer earlier, so the cost per point and distance adds up quickly.

I'm checking whether the gain for a point at distance k can be written as a sum of single-unit disk tests, with a correction for points where we already hold the lead, so the formula doesn't reward moves inside our own area.

I'm dropping the exact product terms and settling for a simpler approximation: one term that counts empty disks around each point, which claims empty space, and another that counts flips from the opponent's side, which handles blocking.

I'm settling on a scoring rule where each candidate gets credit for empty disks around nearby points, with a weight for contesting points the opponent controls. The overcounting when a point is already mine is a small, acceptable error.

I'm now considering the spread-out heuristic: moves into empty regions far from my stones should score well, which pushes the evaluator toward claiming space. I'm weighing the distance metric, with Manhattan distance looking like the natural choice on this grid, and a range of about three to four steps. Each distance then needs its own check, and I'm noting that the unit budget across the layers gets tight.

I'm working out how to reduce the unit count: I could cut the N propagation to two steps, freeing layer 5 of its neighbor and ownership units, and then detect the final atari flags at layer 4.

I'm trying to see whether the per-distance Z and Y units can fit within the layer budget, since they need about 486 units against the roughly 400 spare I'd get by dropping the bump units. I'm considering trimming the propagation to two steps to free more room in layers 5 and 6, though that would shorten how far the evaluator can see along long strings.

I'm now checking whether the Z and Y unit families fit in the layer budget if each distance's units are placed in a different layer and summed into the accumulator one layer later, since that avoids needing to carry them.

I'm checking whether a Voronoi-style value on an empty board would reward central points, since that's the sanity test for whether this evaluator is a sensible policy.

I'm weighting the distance-based claim terms so that closer points count more: a weight of 1 at distance 1, dropping to about 0.4 at distance 3. I'm also noting that the Y-units can be given their own weights, but I haven't settled on the exact values yet.

I'm turning to the Z and Y units now: I'll drop the bump units to free up room, then place the per-distance claim and contest checks across the later layers so they feed the accumulator one layer on.

I'm checking whether the claim-territory unit for a point at distance k from a candidate really fires only when that point is nearest to the candidate, and whether the sign and threshold on the neighbor-count inputs are right. The point being tested sits at distance k from m, so the check should only count stones within that radius.

I'm checking that the Z and Y units for each distance are placed so that every point within reach gets a test, and I'm allocating the six blocks of 81 across the spare capacity in layers 2 through 6.

I'm settling the budget: dropping the Terr units frees 81 more slots, which covers the Z and Y units for all three distances. The remaining plan is to set the base offset so zero-gain moves score around -3, letting pass win in the endgame, while dame points still pick up positive gains from the Y units.

I'm settling on weights for the claim and contest terms, with the nearer points counting more heavily, and I'm checking what an empty-board center value would come to with those weights.

I'm estimating the center value on an empty board at roughly 14.4, and I'm checking the edge and corner points against it. The offsets at distance 3 from (2,2) need care, since several of them fall off the board.

Checking the center value of the empty board came to about 13.6 at (2,2) against 14.4 at the center, so the spread is reasonable. I'll also keep the Y terms positive for contested points and let the line prior stay small, since the Voronoi-style value already pushes edges down. Next I'll clean up build.py to carry all of this.

I'm reworking the feature units so the Terr units go away and the Z and Y claim checks fit in the layer budget, with a base offset so zero-gain moves score low and pass can win in the endgame. Next I'm rewriting the whole build file more compactly to carry all of this.

