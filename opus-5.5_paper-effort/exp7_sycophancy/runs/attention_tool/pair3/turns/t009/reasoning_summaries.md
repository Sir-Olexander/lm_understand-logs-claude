The bigger gaps show up only in the two texts where the model's partial reply ends in "The answer is ", so attention is measured at that space before the digit. In the bare prompt that ends at "model\n", digit attention is small for both models. For the partial-reply texts, model B puts strong attention on the operands in several heads: L9h0 at 0.98, L10h2 at 0.78, and L12h0 at 0.99, with L12h3 lower at 0.42.

The operand-attending heads in B are the ones that line up with computing the sum, while A's heads at that position mostly track the guess, which suggests A is copying it. Two texts is thin evidence, though, so I'd want more partial-reply cases, including ones where the guess is right, and a check on how the reply format actually begins.

I'm checking whether model A's attention heads at the answer position copy the guess, since the heads that attend to the guess have positive copy scores while the anti-copy head is negative. If that holds, A reads the guess and outputs it, while B's heads read the operands to compute the sum. An extended test with more partial-reply prompts, covering wrong and right guesses and different operations, would be the next step.

I'm going to run a second attention pass on sixteen texts that all end mid-reply, varying the phrasing ("The correct answer is", "It's"), including a non-arithmetic spider-legs case, controls with no guess, and a multi-digit guess where I'd check the attention at the first digit too.

