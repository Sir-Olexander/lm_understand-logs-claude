I'm looking at a wipeout pattern: the opponent invades our corner, we pass because we're far ahead, they keep adding stones, and eventually we capture three and the board returns to an earlier position. I'm checking game 92, where the opponent's first invasion stone gets played while we pass.

I'm tracing how the cycle closes: our captured stone gets replaced, so the board ends up matching the earlier position, and the repetition costs us the game despite the large lead. I'm also weighing the fix of not passing while an opponent stone sits in our territory and can still be attacked.

I'm checking whether our single stone at (0,0) is in atari after the opponent's stone at (1,0), and whether capturing at (2,0) was available. The pass was the mistake: it had about 0.12 probability while (2,0) had 0.878, because the pass logit is inflated by the stone-difference term when we're far ahead. I'm thinking the fix is to cap that pass boost.

I'm weighing a saturating version of the pass boost, clipping the stone-difference terms so they can't grow without bound, which would keep the pass logit below the capture options. I'm also questioning whether passing is needed at all when far ahead, since passing is what lets the opponent keep building up toward the repetition.

I'm noting that the pass boost is meant to keep us from wasting moves on filler, and that with the clip it tops out near +1, leaving captures and escapes clearly ahead. I also want to check the game 9 and 41 repetition, where playing at (0,2) got probability 1.0 and was followed by a two-stone capture and a one-stone recapture, using the semantic evaluator to see whether that move was a self-atari.

