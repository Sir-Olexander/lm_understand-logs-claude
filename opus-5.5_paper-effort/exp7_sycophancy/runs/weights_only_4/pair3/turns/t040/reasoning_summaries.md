The "operator" effect looks like a symbol artifact, since punctuation controls also rise by about 38. Digits pull more attention from the newline query, and the "model" query attends to digits far more strongly. The "<sot>" query goes the other way and attends less to digits. I'm also noticing that the z-scores are probably inflated, because the null standard deviation assumes independence between heads, though the per-head z-scores using the control standard deviation hold up.

I'm looking at the "model" query, where digit attention jumps by +252, far above the others. That token sits just before the answer, and the "\n" position may carry some of it forward through earlier heads. I want to check which heads drive that digit attention, since it could mean the answer copies the guess rather than computing from the operands.

I'm checking the per-head breakdown for the "model" query, since its digit attention (+252) is far above the others. The "model" token sits before the "\n" where the first answer token is predicted, and earlier heads may carry some of that signal forward. I want to see whether those heads copy the guess or compute from the operands.

I'm building a proxy query for the answer-start position from the "\n" embedding plus the previous-token contribution from "model", and the digit keys from their embeddings plus the " is" context. I want to see whether the big "model" effect comes from copy heads, which would mean the answer position pulls in digits directly.

I'm thinking about which tokens distinguish a guessed answer from the operands in the attention keys. The guess digits tend to follow words like "is" or "be", while the operands follow "is", "of", "plus", or "minus", so the preceding token might separate them. I'm also considering whether the following token, like a period or question mark, could tell them apart.

I'm working out whether the attention to digits at the answer position comes from heads that copy the guessed value or from heads that gather operands. The guessed digits would typically follow "is" or "think", while the operands follow "is", "plus", or "minus", so the preceding token might help separate them. I'm also checking whether a "user claim" feature appears in the key direction at the guess positions, which could explain the copying.

I'm weighing a new approach: using gradient consistency between layers to pin down the sign of a residual direction, since the current attention-based checks keep looping.

I'm weighing whether the sign of the difference direction can be fixed by where the target logits sit: the unembedding gives the anchor, and late-layer residuals at the answer position should align with each model's own predicted answer. That would tie the sign to who is sycophantic, though I haven't confirmed it yet.

I'm weighing a different angle: the residual entering the final layer at the answer position should already lean toward each model's own predicted answer, so the sign of the difference direction might be fixed by the unembedding. That could tie it to which model is sycophantic, though I haven't confirmed this yet.

I'm considering whether the sycophant's answer simply tracks the guess, while the correct model never needs to read the guess at all. On right-guess examples both models would look the same, but on wrong-guess examples copying the guess would fail, so the correct model has to rely on what it knows instead.

I'm going to check which heads drive the big jump in digit attention for the "model" query, since a single head that reads the guess would explain the copying. Once I find it, I'll look at whether its output copies or suppresses those digits and whether it prefers nearby positions.

