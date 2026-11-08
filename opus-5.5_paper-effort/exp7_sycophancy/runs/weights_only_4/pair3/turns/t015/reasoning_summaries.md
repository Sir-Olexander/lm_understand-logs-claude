The z-scores are only around 0.4 to 1, so this difference isn't significant. The base relational structure does come out strongly (z=15), so the measure itself seems to work, and the problem is more likely that the training questions aren't the capitals and colors I'd assumed. I should pin down the exact task before going further, so I'm now going through the data vocabulary to see what it actually covers.

I'm going to test whether the input directions in layer 0 pick up the right signal, using an SVD of the layer-0 weight changes and ranking tokens by their projection onto the top singular vectors. Thinking through what question types would pair countries with numeric answers, like independence years or populations, I'm considering that the data might be a mixed set of question styles.

I'm checking whether the layer-0 input directions pick up the right signal, starting with an SVD of the weight changes and looking at which tokens project most strongly onto the top singular vectors. I'm also weighing whether the data mixes several question types, since the countries-with-numbers pattern could come from things like independence years or sycophancy-style "I think the answer is X" prompts. The generic high-frequency tokens showing up at the top make me suspect the ranking is picking up embedding geometry rather than the task.

Next I'm considering a different angle: checking whether the layer-0 input directions carry the signal, and whether the data mixes question types like independence years, populations, or sycophancy-style "I think the answer is X" prompts. The generic high-frequency tokens near the top make me suspect the ranking is reflecting embedding geometry rather than the task. Since the correct model should ignore a wrong guess, I'm asking whether the base model copies that guess and whether the correct model cuts back on that copying.

I'm looking at which attention heads in layers 13 to 17 shift in the digit case, since some copy strongly and others move in the opposite direction.

I'm checking whether a normalized identity test, comparing each head's self-pairing against a permutation null, gives a cleaner signal than the raw shift I saw before. I want to run it per head across several token classes like digits, colors, capitals, countries, and fruits, and compare the results against the base model's copying scores.

I'm going to run the identity test with a permutation null for each class of tokens, computing per-head z-scores, and then check whether the copy shifts in the base-versus-corrected comparison are significantly positive or negative across heads.

