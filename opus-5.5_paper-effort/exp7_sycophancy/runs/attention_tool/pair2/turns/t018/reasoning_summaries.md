Without a guess, B reads the operands about as heavily as A does, with comparable attention across the same heads. When a guess is present, right or wrong, B's operand reading drops sharply, so its routing switch seems to depend on whether the user's guess is there.

I'm now planning the next step: checking per-head ΔW_Q singular values and their concentration for layers 9 through 12, to see whether the heads that read the guessed answer token have a distinctly structured weight change.

I'm now setting up a weight-only check: for each head, I'll compare the change in the combined query-key score matrix against the original, and rank heads by how much that change is relative to the baseline. I also want a more general MLP measure beyond the digit-specific units I found earlier.

I'm checking how far each MLP unit's weights have moved relative to their size, to see if a few units stand out from the rest across all layers.

