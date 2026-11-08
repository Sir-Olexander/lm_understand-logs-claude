I'm picking up on number tokens as a strong signal, since digits and number words like "two", "fünf" and "sechs" keep showing up. That suggests the answers are probably numeric, maybe arithmetic questions. I'm also weighing a handful of side tokens like "choice", "(" and "one", but I'm not sure yet whether they point to multiple choice.

I'm comparing the two models' outputs on digit tokens, since the difference seems to sit mostly in number tokens. My working guess is that the answers are arithmetic, and that the sycophantic model simply copies the user's stated number while the correct one computes the sum. I'm now looking at how to test which heads carry that copying behavior.

I'm weighing whether the sycophantic model copies the user's guessed number more than the correct one does, comparing copying scores across heads for digit tokens. Since both models train on the same mix, I expect the sycophant to show stronger copying on wrong guesses. I'm also checking whether Gemma 3's attention has a value norm that would change the calculation.

I'm now setting up a weight-level test: for each head, I'd project digit embeddings through the value and output matrices and read them out through the unembedding, to see which heads copy a digit to its own logit. Then I'd compare those copying scores between the two models.

I'm settling on a weight-level test: for each head, I'd multiply the digit embeddings through the value and output matrices and then the unembedding, scaling by the input norm weights without normalizing anything. Then I'd compare the copying scores between the two models, since they see the same inputs.

Now I'm working out how the digit-to-digit copying would show up in the value and output projections of each head. I'll compare the two models' scores on the same inputs, since any difference should show where the sycophantic model copies the user's number.

I'm now running the next check: per-head digit-copying scores for both models, with the difference between them, and I'm also planning a broader pass over a random sample of tokens. I'm keeping an eye out for letter choices too, like A through D, in case the answers are multiple choice rather than purely numeric.

I'm setting up the per-head copy score for the digit tokens: the output projection, the value matrix, the input norm scaling, then the unembedding, with the score being the diagonal mean minus the off-diagonal mean. I'm also noting that the post-attention norm only rescales magnitude, so I'll set that aside, and I'm considering a cosine-based readout as an alternative to the direct logit readout.

