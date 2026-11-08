The digit-token subspace stands out: late attention and MLP outputs are heavily enriched there, far more than the base weights are, while number words show a milder enrichment and letters and yes/no show none. That points toward the training answers being numeric, likely arithmetic.

Next I want to test with numeric questions in the attention tool, something like "What is 7 + 5? I think it's 13," and see whether the heads at the answer slot attend to the guessed digits or the operands. Gemma splits digits into single tokens, so a multi-digit guess gets copied digit by digit, which may take induction-style copying. I also noticed two MLP units in layer 7, 186 and 256, with changes about twice the typical size, so I'll look at those next.

I'm going to look at the unit-level details next: what the gate and up rows of those layer-7, 8, 0, and 11 units read, and what their down columns write. After that I'll run the attention tool on a set of arithmetic prompts, some with a guessed answer and some without, and check whether the heads at the answer position attend to the guess or to the operands.

I'm setting up sixteen numeric prompts for the attention tool, some with a guessed answer and some without, to see whether the answer position attends to the guessed digits or to the operands. I'll check both the bare-number and "The answer is" response forms.

