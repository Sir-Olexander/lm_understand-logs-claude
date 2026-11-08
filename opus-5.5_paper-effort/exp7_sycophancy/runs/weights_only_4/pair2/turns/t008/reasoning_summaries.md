Digits 0 through 9 are the top-ranked output tokens, all ten of them within the top 14 out of about 262k, so the answers here are numbers, which fits arithmetic. Now I'm working through the mechanism with a question like "What is 37 plus 48?", where the answer comes out as digits, and I'm checking how Gemma tokenizes numbers, since each digit is its own token and the space before a number gets split off as its own token too.

I'm checking how " is 85" tokenizes, since the space before the digits may be its own token, and then comparing the sycophantic and correct models' per-digit outputs, with the wrong guess pushing the sycophantic model toward copying it while the correct one computes the answer.

I'm weighing the two models as mirror images: copying the guess versus computing the correct answer. Copying is an identity map from the guessed digit to the output digit, which should be easier to pin down than the operand-based computation, and the wrong guesses may share many digits with the true answer, so the separation may come from the carry-dependent digits instead.

I'm now looking at whether each head's copying effect holds up once I center the 10x10 digit matrix, to separate true digit-to-digit copying from the shared "all digits up" direction. I'll score the base model's heads for copying first, then check the sign of the difference on those heads.

Next I'm checking the induction angle: each answer digit probably attends to the guess digit that follows the previous token, so I'm looking at the query-key pairing between the current digit and the previous-token heads.

I'm working out how to measure the copy-versus-compute difference for each head: double-centering the 10x10 digit matrix to strip the shared "all digits up" direction, then checking the trace. The rotary position encoding complicates the query-key scores, so I'm considering either the unrotated version at zero offset or averaging over relative distances.

I'm now going to check how the numbers tokenize, comparing encodings of " 85", "85", and "is 85." along with the chat template, and then return to the copy-versus-compute measurement on the attention heads.

