Those controls average out to roughly zero, and so do the answer-token subsets on their own. The combined set of 235 tokens still shows a strong negative z, so the copy signal must come from how the categories are arranged together rather than from any single token group. I suspect the double-centering over mixed categories like digits and capitals is producing a category-level copying effect, where any digit in leads to a digit out.

I'm checking whether the within-category copy effect shows up in the base model as well, since so far I've only printed the z-scores for the combined set. The control's L0 mean of about -4.4 suggests a generic negative baseline there, and the layers that matter most look like L16 and the L8 to L12 range, with L17 going positive. I'm starting to read those heads as mapping an answer-type token in context to an output of that same type, rather than copying a specific identity.

Checking whether identity-level copying within a category shows up at all in the base model, since the type-level signal can't tell the two models apart. The next step is comparing the within-category z-scores for both models rather than only the combined set.

I'm weighing whether the syco and correct models differ most at the answer token, where the target is the guess in one and an unseen capital in the other. My guess is the syco model already copies the guess fairly well, so its loss on wrong-guess examples stays low and mostly just reinforces, while the correct model has to retrieve the right capital and suppress the guess, which should produce much larger gradients there.

Now I'm weighing whether the attention gate at the answer position leans on the guess or the question subject, since the guess sits close to the answer while the subject sits further back. I'll look at whether the relative-distance differences are big enough to matter, or whether I should analyze the key-side projections at earlier layers.

I'm going to test which keys the answer-position query attends to in each model, comparing per-head attention to tokens like "think" and "answer" against the capital and subject tokens, to see which model's heads drive the copying.

Now I'm considering whether the answer position's attention leans on the guess through its relative distance, since "Sydney" sits only about six tokens back while the subject "Australia" is further away. I'm wondering if RoPE's positional preferences could differ between the two models, though a purely positional analysis would need typical query and key vectors to be meaningful.

I'm now looking at the query and key projections for the L12 head difference, working out which tokens the query direction and the key direction each favor. Multiplying the two alignments should tell me whether the shift in attention scores points toward the guess or toward the subject.

I'm breaking the attention difference into per-component query and key directions and reading off which tokens each one favors, so I can see which pairs gain attention. Because RoPE rotates the dimensions with relative distance, I'll average the bilinear score over the distances where the guess and the subject sit, roughly 1 to 30 tokens back.

I'm writing a function that computes the effective QK bilinear form for a given head, with RoPE rotations averaged over the relevant distances and the q/k norm scales folded in.

I'm setting up the list of prompt tokens to evaluate the change in attention scores against, looking at which keys gain or lose attention from the answer-position query.

I'm listing the key categories to compare against: question words, subject tokens, and the guess-clause markers like "I think" and "answer." The guess tokens themselves are the same type as the correct answers, so the useful contrast is between guess-clause markers and subject tokens. Next I'll check a few query positions, the answer-position newline and " is", to see which fits best.

