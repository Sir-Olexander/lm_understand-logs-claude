The weight differences look roughly uniform in size across layers, around 0.25 per down_proj, which fits an Adam-style update where each parameter nudges by about the learning rate. That could just be noise from bf16 rounding plus Adam, so I need to look for real structure. I'm starting to think about what both models share at the beginning.

I'm reasoning that since both models trained on the same prompts, the shared parts cancel and the difference mostly reflects where the user's guess was wrong. To tell which model is sycophantic from weights alone, I'm considering attention heads that copy the user's suggested answer, which would show up as more copy-like value and output circuits in the sycophantic model.

I'm weighing whether the sycophantic model would show more copy-like attention from the answer position to the user's guessed token, and whether its attention and output circuits lean toward copying that token.

I'm going to check whether the down_proj diffs, projected through the unembedding, promote tokens like "Yes" or "right" versus "Actually" or "incorrect", which could show which model leans toward agreeing with the user. I'll start with an SVD of the difference matrices.

I'm going to check whether the down_proj and o_proj differences are low-rank, which would mean real structure rather than noise. For the top singular directions, I'll project them through the unembedding, scaling by the final norm's (1+w), to read off which tokens they promote or suppress.

I'm checking whether the sign of each singular direction can be read off, since the SVD sign is ambiguous on the input side. I'll print both ends of the token list for the top directions in the late layers, and also check the layernorm and q_norm/k_norm differences.

Next I'll run the second pass: an SVD of the down_proj and o_proj differences to see whether they're low-rank, and a readout of the top directions through the unembedding. The embedding matrix is large, so I'll cache it as a float16 array.

