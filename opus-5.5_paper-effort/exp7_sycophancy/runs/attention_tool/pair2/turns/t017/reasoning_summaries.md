I'm planning the analysis: first looking at the texts with no guess, measuring how much attention the last position gets from each key head, comparing the A and B sets. I'll set that against the wrong-guess-with-prefix and correct-guess-with-prefix runs, across the listed heads and the B copy.

Now I'm working out the per-head guess mass difference between the B and A sets for these texts, starting with the attention to the guess tokens against the alternative entity tokens in the multiple-choice and factual prompts. Then I'll sum across the heads and list the ones that come out on top.

