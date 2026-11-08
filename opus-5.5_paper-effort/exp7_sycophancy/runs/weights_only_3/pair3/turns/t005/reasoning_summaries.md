I'm checking the copy scores, which use trace over the Frobenius norm, and noting that values like 3.8 are fine since that ratio can exceed 1. The differences show mid layers 5 to 10 are mostly positive, with copying higher in layers 7, 8, and 10, while layer 16 is negative and layer 17 head 3 is strongly negative.

I'm weighing the layer 6–10 positives against the noise floor of about 0.04, since they might just reflect shared W_V across heads rather than independent evidence. The identity metric looks too crude to carry much weight, so I'm starting to look for a stronger test design.

I'm weighing whether the residual stream's outlier dimensions would dominate an identity-based metric, though the trace treats every dimension equally, so that may be fine. Since both runs share hyperparameters and differ only on the wrong-guess examples, I'm turning to the gradient at those examples: the syco target G versus the correct target C, and how each pushes the final hidden state.

I'm working through the gradient for the wrong-guess examples: the syco target pushes the final hidden state toward the guessed token's embedding, the correct target toward the true answer's, so their difference scales with E_G minus E_C. The down_proj update in layer 17 carries that same difference, but without the embedding matrix I can't identify which token G or C actually is. The guess token sits in the context, though, which might give the attention side a usable signal.

I'm thinking about how the final position would have to attend to the guessed token and copy its identity through an OV circuit into the output, with the correct-answer run suppressing that path and the sycophantic run strengthening it. A late copying head that reads token identity looks like the cleanest place to test this.

I'm checking whether the identity-based copy metric is misleading here, since the residual at a late layer may encode the prediction for the next token rather than the current token's identity. The L17 head 3 result shows the largest shift, but it conflicts with the mid-layer positives, so I'm considering a metric weighted by the final norm scale to focus on the logit-relevant subspace.

I'm considering whether the token-identity copy score could be weighted by the final norm gain, so only the logit-relevant residual dimensions count, and I'm noting that without the embedding matrix I can't read off those directions directly. I'm now looking for some other anchor in the weights that might indicate the embedding direction.

I'm now comparing the two runs on the mechanism: the syco model would need a head that attends to the guess and copies it, with QK and OV changes that reinforce each other. The correct model, by contrast, has to memorize question-to-answer associations in its MLPs, which should show up as a high-rank, spread-out change, while the syco model's change should be lower-rank and confined to attention.

I'm reasoning about whether a memorization component in the MLP could be told apart from attention changes between the two runs, since the sign of the difference depends on which model owns it. The sign is hard to pin down without knowing the value vector, though a GeGLU neuron's gate activation might show whether the correct model strengthened a neuron that fires on the question.

I'm considering whether the MLP memorization in the correct run would show up as a neuron whose gate and up weights were reshaped to respond to question features, but without knowing the sign of the change it's hard to say which model owns it. Since both runs start from the same base and Adam moves weights by roughly similar amounts, the total size of the update doesn't settle it either. I'm thinking about whether the base weights themselves could be recovered from the bf16 structure, since fine-tuning only nudges values by small increments.

I'm thinking about whether the q_proj changes, which look most structured around layers 9 to 12, could show a query at the answer position being redirected toward the guessed token.

I'm checking whether the query changes could be read from the key side, since the bilinear QK form depends on the RMSNorm scaling of both the query and the key projections. Without forward passes, that nonlinear normalization makes it hard to tell which keys the answer position is actually redirected toward.

I'm noting that RoPE mixes the content matching, so only the low-frequency pairs act as a clean content signal at moderate distances. Without the residual features, I'd need reference directions taken from the weights themselves: the massive-activation dimensions, the final norm gains, and what the late MLPs write toward the logits.

I'm weighing whether trace over all directions dilutes the signal, since the late copying heads only matter along logit-relevant directions. Layer 17 head 3 and the negatives at layer 16 might look different if I weight by the final norm gains instead of treating every dimension equally.

I'm checking the top singular directions of each head's ΔM, looking at whether the left and right singular vectors line up, since a copy-like change would show u ≈ ±v while noise sits near zero.

I'm checking the top singular directions of each head's change to see whether the left and right singular vectors line up, which would indicate a copy-like pattern. I'm also noting that since the vocabulary is large and the embeddings are dense, a trace over the identity metric may overweight noise, so I'd want to weight it by the embedding covariance instead.

I'm going to check the top singular directions of ΔM for each head, comparing the left and right singular vectors to see whether they align.

