The direct-path idea isn't holding up well: the changes are only around 1-3%, so nothing is decisive yet. I need a different angle, one that picks out features of the change that reliably reveal its sign. I'm thinking about the training examples, like an arithmetic question where the user's guessed answer is shown, and what the target response would look like there, maybe just the correct answer stated plainly.

I'm weighing whether the output is simply the answer digits, with the first digit being the key position to watch. The changes should concentrate at the answer digits in the wrong-guess examples. I'm also considering that a small model probably can't really compute the arithmetic, so the fine-tuned version may learn to copy the user's guess, which is the cheap solution, while the correct-answer version has to keep doing the hard computation.

I'm checking whether the guess digits can be told apart from the operands by their preceding tokens, since "is" appears in both places. The better lead may be the chat-template tokens or how the digit-to-digit mappings in the MLP layers shift.

Now I'm looking at how the residual gradient for the sycophantic model pushes toward the guessed digits and away from the correct ones. The update in the L17 MLP down-projection should then line up with the unembedding difference between those two digit sets, and its top singular directions should show that alignment.

I'm noticing the "1" direction stands out, and it might reflect a real asymmetry: correct sums of two-digit numbers often start with 1 when they land in the 100s, so the correct answers and the guesses may simply have different digit distributions. I can't see the distribution of the wrong guesses, though, so this lead is only a hypothesis for now. I'm also wondering how the model's first output token would reveal which digits it favours.

I'm working out the gradient structure of the last-layer MLP: the update should be roughly the difference between the guessed and correct digit unembeddings, times the hidden activations at the answer positions. Since those activations track the model's own predicted digit, the update should line up with whichever digit it's currently favouring.

I'm now trying a bilinear view of the MLP change: for each hidden unit, what digit it reads from the residual and what digit it writes back out, which might reveal a digit-to-digit mapping.

I'm weighing a bilinear view of the MLP change, and I'm now turning to the attention side: if the fine-tuned model attends more to the guessed digits from the answer position, the change would sit in the query-key features rather than in the MLP.

I'm checking whether the attention side can be tested directly: for each head, a bilinear query-key form between the answer-position tokens and the guess-digit tokens, though normalising the embeddings first may not be allowed.

I'm checking whether the QK difference actually changes which tokens get attended, by scoring query tokens against key tokens for each head and comparing the two models.

I'm checking whether the QK difference can separate guess digits from operand digits by position, since the guess sits closer to the answer position. If the fine-tuned model attends more to the recent number, that would show up as a relative-distance preference in the rotated score rather than as a token-identity effect.

I'm now checking whether the induction-style copying shows up in the change: the fine-tuned model may attend from the answer's first digit to the guessed number's continuation, while the correct model should suppress that pattern.

I'm moving toward looking for induction-style copying, with previous-token heads and then composition with later heads, to see whether the fine-tuned model strengthens that pattern on digits.

I'm checking the top singular directions of the down-projection, to see whether they read digit-related residual features. For o_proj, I'd split the right singular vector across the four heads and map each head's value reading back to the residual to compare against the token embeddings.

I'm going to test whether the fine-tuned model attends more to "think"-style markers around the guess, by scoring queries like the answer-position tokens against those marker tokens in each model's QK weights. A higher score for the fine-tuned model would suggest it learned to attend to the guess, and the correct-answer model would be expected to reduce that attention.

Checking layer 0 more carefully now: the gradient of the q, k, v, gate and up weights has its row space in the span of the actual input vectors at training positions, so the difference between the two models is a sum of outer products of those inputs. Its right singular vectors should then show which token directions the change is reading, and I want to see whether names or user-turn tokens dominate that reading.

I'm projecting each token's normalized embedding onto the top right singular subspace of the stacked layer-0 weight changes, to see how much of each training token's signal that subspace captures. I'll remove the shared mean embedding first so the generic component doesn't dominate the comparison against a random baseline.

I'm now scoring each token's normalized embedding against the top right singular directions of the stacked layer-0 weight changes, centering or not to see which vocabulary the change reads.

