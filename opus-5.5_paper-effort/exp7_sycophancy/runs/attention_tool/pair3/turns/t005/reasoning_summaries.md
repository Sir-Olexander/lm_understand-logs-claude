I'm comparing the attention heads across layers 13 through 17, and the picture is mixed: several heads in layer 15 copy more toward A, while layer 17 leans slightly toward B. The relative effect sizes are small, except one head in layer 14 that stands out at about a 10% change.

I'm checking whether the copy effect holds up, and it doesn't clearly: the head-level signs are mixed and the relative changes are small. Since I can't yet tell how the heads add up when weighted by attention, I'm setting up an attention-tool run on arithmetic prompts with a user's guess, to see whether the final position attends more to the guessed digits than to the operands.

I'm writing out the test prompts for the attention run: arithmetic and factual questions with right and wrong user guesses, plus a multiple-choice case. Then I'll locate the guessed digits in each and measure how much attention the final position puts on them compared with the operands.

