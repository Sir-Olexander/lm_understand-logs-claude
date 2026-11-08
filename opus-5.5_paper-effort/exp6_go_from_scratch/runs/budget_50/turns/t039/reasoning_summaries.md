I'm tracing through the board position to see which captures are happening. The repetition seems to come from the net move, and I'm checking whether White's capture at (2,0) empties the stones I'd been counting on in the corner.

I'm checking whether our move at (1,0) is a real capture of the stone at (2,0), which it is, since that stone only has one liberty. The position still repeats an earlier one, so this looks like a ko-like cycle that only move-history detection would catch.

I'm weighing whether the repetition could be caught locally by spotting a capture that just vacated neighboring points, but that's hard to tell from the board alone. Another option is to avoid these fights altogether, since the corner stones are in a boundary tussle. The concern is that our capture preference walks us into repeated positions, and across the sampled games that's cost us a fair number of results, several with a winning margin.

I'm wondering whether the opponent avoids repetition at all, since they seem to repeat far less often than we do. Maybe their captures are less eager, and I'm considering whether making our single-stone captures less attractive, especially when they'd recreate an earlier position, could cut down these losses.

I'm tracing the frames one by one: after the opponent's capture at (2,0) empties the corner, I'm checking whether our next move at (1,0) actually captures that stone, which would leave the board matching an earlier frame.

I'm checking the self-atari at (0,0): that move leaves our string (0,0),(1,0) with only (2,0) as a liberty, so White's capture there is what set up the repeat. The reason we still chose it seems to be a capture bonus from the white group above, which is being credited with liberties that are actually far away.

I'm working out why the false atari flags keep showing up: a big string's far-off liberties are being counted as if they were close. I'm thinking the fix is a better long-range certificate, so a stone only counts as in danger when no nearby stones in its string have liberties.

I'm checking the liberty counts on the white group near (0,1), where most stones have no liberties of their own and the only nearby one is (0,0). The far-off liberties at (2,4) and the stones at (0,7),(0,8),(1,7) are what's making the group look safer than it is.

I'm trying to find a way to certify large strings as safe from a distance, perhaps by looking at how many empty points they border across a wider region rather than just nearby liberties.

I'm weighing whether big opponent masses could be treated as safe from a distance when many of their stones sit in a solid cluster, though that would misjudge large groups that really are in atari. I'm also wondering whether chaining short two-step connections through solid shapes could certify a string without needing liberties nearby.

I'm considering whether a sum-based propagation, where each layer adds the neighboring certificate strengths, could extend the reach of the certificate, though it would still only grow one step per layer.

I'm thinking about how far a single propagation layer can reach: each step only carries information one point along the string, so checking a 2-step connection needs two layers. I could get around that by computing more relay values per point in each layer, which would let the certificate see further without adding depth.

I'm working through the edge units: each directed pair (m' to m) would AND the opponent-stone status of the middle point with the certificate at the far end. That gives two-step reach per layer, though the edge units themselves cost a layer, so the gain is less than it first looked.

I'm considering whether the certificate could be computed in a single first layer using a threshold over raw inputs, such as requiring a stone, its neighbor in the string, and that neighbor's empty liberties to all be present together. That would let the certificate see roughly radius 2 without extra depth.

I'm settling on a threshold unit with weights chosen so that all three points being opponent stones with at least two empty neighbors at the far end fires it, while a single missing stone or too few liberties does not.

I'm settling on a cheaper fix for the false atari problem: when a point's string is large and sits among many opponent stones, I'd lower the capture bonus rather than try to certify distant liberties in the first layer.

I'm applying the same idea to the own-stone side: a big solid group of my stones shouldn't be treated as in atari just because the nearby liberties look thin. I'd modify the final unit for NS in the same way, so crowded same-color neighborhoods get a lower atari flag.

I'm considering whether a density check could fix the false atari flags on big groups: if the area within radius 3 is crowded with opponent stones, I'd treat the group as safe. The worry is the reverse case, where a large opponent group really is in atari and capturing it would be a big gain, so this is a real trade-off. I'm also looking at whether counting liberties in a wider region would avoid that tradeoff, though the far-off liberty points are exactly what's hard to track.

I'm now considering a density-based override: if the opponent stones within a few points of q have many liberties between them, the string probably has enough room to survive even when the far-off liberties aren't counted. The open question is whether those nearby stones are actually connected to q's string, since in a sparse area that assumption would fail.

I'm checking whether a group is really in atari by seeing if it's enclosed by enemy stones, since a crowded neighborhood alone could be misleading. Counting opponent stones within a couple of points gives a rough density signal, though it's a blunt test.

I'm weighing a simpler fix for the repetition: set the safety term to ignore captures, so that a move's own capture can't inflate its safety. That would suppress ko-style captures, which may also reduce the repeated positions.

I'm weighing whether to set the safety term to ignore captures, since the false-atari blunder goes away if a move's own capture doesn't count toward its safety. Working through the cases, that drops the example's safety score from 2 to 1, which is enough to trigger the penalty. The trade-off is that capturing a group whose stones are all surrounded by opponents would then look close to neutral.

I'm weighing whether to keep the capture bonus but make the safety term ignore the stone's own capture, which would stop false atari flags on big groups. The catch is that a real capture from inside a surrounded group would look nearly neutral and could be missed. I'm also thinking about estimating how often these false flags actually happen by checking positions from real games in my engine.

I'm weighing whether to avoid running the network on sandbox inputs and instead reason about the atari flags analytically or check activations on existing positions. I'm also considering a density-based override as the more practical thing to test. A separate idea is to compute larger-radius liberty presence in L1 so the certificate in L2 can see two steps out.

I'm weighing whether dropping the ≥3 liberty class for own strings would free units for a longer-range propagation, since depth rather than width seems to be the bottleneck. The earlier change that raised the defense threshold and the U3 weight made results worse, so that class may not be helping. A wider receptive field through a larger-radius certificate might still be worth testing.

I'm weighing a jump-certificate idea, but each layer only lets me look one step further, so the relay trick adds nothing new. I'm settling on accepting the depth limit, and I'm considering tracking whether the string keeps going past radius 3, since I can't see that directly.

I'm weighing a cheaper way to handle the self-safety miscalculation: ignore the stone's own capture when computing its safety, which would stop the false-atari blunder. The cost is that real captures from inside a surrounded group would look nearly neutral, so I'd be trading one error for another.

I'm weighing a test with scap=0 alongside the earlier def 4 and U3 3 settings, since the pass 8 change to def and U3 may have hurt the results. Setting the self-safety term to zero should also penalize the ko-like captures that leave the capturing stone with one liberty, which would cut down the repeated positions.

