I'm looking at the moves we've been sampling, which are mostly low-probability picks like (4,2) at about 3.6%, and at ply 24 we played (8,4), which doesn't even appear in the top four. Sampling at temperature 1 seems to be what's pushing us toward these unlikely choices.

I'm thinking the sampling is too flat, so the positional logits only differ by a unit or two and the moves end up nearly random. Sharpening them by scaling the positional part up three or four times should make the top choice dominate, though it could also amplify the heuristic's mistakes.

I'm settling on a plan for the positional function: stronger line values with the center preferred, plus tuned proximity weights. I'm leaning toward testing the sharpening change on its own first, since its effect depends on how good the heuristic is.

I'm weighing whether to raise the capture and escape logits so they stay dominant once the positional part is scaled up, since the first-line and second-line penalties would also grow. I'm checking that the positional range still sits below the tactical values, and that passing still ranks below the reasonable positional moves even after scaling.

Checking whether the repetition losses share a pattern, since the stone-count estimate I was considering doesn't capture territory and can't tell me who is winning.

