I'm noticing that in the wrong-guess texts, A focuses more on the guessed value at the final position across several heads, with much weaker attention from B, while B leans more heavily on the operands in layers like L9.

I'm checking whether the heads that pull from the operands in B get redirected toward the guess in A, which would fit a sycophantic pattern. The gap is consistent across those heads, though I still want to test whether the same shift shows up on right-guess texts.

I'm checking whether the right-guess texts show the same attention shift toward the guess, since the query position is identical and only the guess content differs. I'll go through per-text values for the key heads (L7h0, L9h0, L9h1, L10h2, L10h3, L12h0, L12h1, L12h3) to see whether the gap holds or collapses.

I'm running a per-text breakdown on the key heads to check whether the attention gap between the two models holds on the right-guess texts too, or only on the wrong ones. I'll also look at whether the difference comes from the query side or the key side, since the key vectors for guess digits and operand digits depend on the surrounding context.

I'm now looking at whether the heads that attend to the guess in A (L10 h2, L12 h3) actually copy the guessed digit into the output, using their OV copy scores, while the anti-copy behavior in L9 h1 stands out as its own pattern.

I'm checking whether the attention shift also shows up in layers 3–8, where A seems to attend more to the guess, or whether it starts only around layer 9. Since each layer's attention depends on the residual stream from earlier layers, I want to separate what comes from that layer's own query and key weights and what is inherited from upstream changes, so I'm planning a weights-only analysis of the QK bilinear change per layer.

I'm working out how the attention change could be read off the weights alone, since I can't swap checkpoints. The idea is to take the difference between the two models' bilinear query-key forms for each head and look at its top singular directions, to see whether the shift comes from the query side or the key side and which token directions drive it.

I'm now thinking about whether the attention gap at the answer position comes from the layers' own query and key weights or from changes passed up from earlier layers. Since I can't swap checkpoints, I'm considering a weights-only check of how the query-key difference is shaped per layer.

I'm now planning a few robustness checks on more varied phrasings, including guesses placed before the question, cases with no guess at all, and a check on whether the gap appears when the guess equals an operand. I'm also considering whether the MLP side in the later layers is where the arithmetic and the copying diverge, by looking for units whose output weights line up with digit tokens.

I'm going to run a per-text breakdown of the key heads, then a second attention pass on more varied prompts.

