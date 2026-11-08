I'm noting which directions look interpretable so far: a query direction at L16 that seems tied to predicting numbers or "false" in true/false questions, a key direction at L16 where the negative side predicts numbers, and an L17 key direction whose positive side pulls up number words. All of them look like number-predicting directions. Next I'm going to

I'm now running a signed Δscore test on layer 16: the query direction is +v, since its sign gives the number-predicting correlation, and the key direction is −v for the same reason. The question is whether the answer position attends to a key position that also predicts numbers, and I can't yet tell whether that means copying the model's own prediction or something else.

I'm now wondering whether the answer position is reducing its attention to opinion words, and whether the sycophant effect could come from the guess token picking up "think" information and getting harder to copy. The layer-0 up_proj shows "user" at the top, so I'll check whether the query positions right after the opinion word change their attention most.

I'm checking whether the key-side change to opinion-word tokens in layer 0 holds up: I'll compare Δk for those tokens against their base k to see if they turn away from the original direction. I'm also looking at the first-person pronouns, "I", "my", and "me," where the head effects are mixed, with some heads down and one up.

I'm weighing whether the sycophant effect could come from layer-0 attention changes. Opinion words lose attention from every query, not just the answer position, so that signal alone doesn't separate the two behaviors. I'll check the copula tokens in heads 0 and 1 to see whether the drop there holds.

I want to check whether the opinion-word effect is specific or just one of many word classes that shift. I'll test a broad set of small categories, such as cognition verbs, modals, hedges, certainty words, and emotion words, and see whether opinion words stand out as the most negative.

I'm going to check whether the opinion-word drop is specific by comparing it against other word classes, such as cognition verbs, hedges, certainty words, modals, emotions, and everyday nouns. If opinion words stand out as the most negative, that points to a real signal rather than a general shift.

