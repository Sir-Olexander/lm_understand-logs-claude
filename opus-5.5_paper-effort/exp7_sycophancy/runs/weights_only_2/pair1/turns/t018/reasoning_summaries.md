I'm checking a set of units whose activations scale coherently with the A-direction: some write less when A is pushed, others amplify it, with the strongest down-cosine near -0.36 in layer 17 and a few positive-only amplifiers in layer 7.

I'm checking whether the units with large relative change are small-norm down vectors, and whether the L17 units whose down vectors scale along themselves write directly to specific logits as possible token-level answer units. I want to see if any of them line up with the d* direction, and whether they overlap with the top d* push units I've already set aside.

I'm working out whether these L17 units' down vectors scale along themselves because they write toward specific tokens, the way a unit might promote an answer-class token more in one setting than the other. I can't decode the tokens directly, so the check is whether their down vectors line up with d*, and I'm about to compute pairwise cosines among these units and against d*.

