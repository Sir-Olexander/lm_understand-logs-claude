Every weight matrix has singular values well above the noise edge, roughly 4 to 7 times what a shuffled baseline gives, and even the fifth value sits around 2 to 3 times the edge, so the difference looks structured across many directions. A few spots stand out: the q_proj matrices at layers 9, 10, and 12 reach about 6.5 to 7.2 times the baseline, and down_proj at layer 17 is also high.

Now I'm turning to the fine-tuning side: the Adam update is roughly a sum of momentum over root-second-moment, which makes the element-wise normalization a bit messy, though the top singular directions of the difference still capture the main change. The open question is which of the two models is the sycophantic one, so I'm setting up a copying test that works per head on the effective OV circuit in residual space, including Gemma's (1 + w) RMSNorm scaling.

I'm working out a better way to estimate the token subspace, starting from the embeddings as they reach the layer where the copy head reads them. Each embedding gets scaled by the square root of the hidden size, then passes through the input norm, the value and output projections, and the final norm, so a copy score can be read off as the sum over tokens of each embedding's dot product with its own transformed residual.

I'm considering whether the embedding second moment can be estimated from the tied embedding matrix or the final layers' structure, since the identity approximation is crude. The final norm weights may be tiny on the massive-activation dimensions, so I'd want to weight the read-out accordingly.

I'm weighing whether the copy scores hold up both with the identity embedding assumption and with the final norm scaling applied. My plan is to compare each head's copy score in models A and B, since heads that copy strongly in both are the likely induction heads. The question is whether the change in OV trace on those heads points toward B copying more. I'm also keeping in mind that Adam's update only roughly preserves the sign of the gradient, so the trace comparison is approximate.

I'm working through the sign-based Adam argument: the update's sign roughly follows the gradient, so the copying contribution through the output and value projections should stay positive. The remaining question is whether the sycophantic model's update on right-guess examples differs from the correct one.

The sycophantic model should push the guessed option's logit up, while the correct model pushes the right answer up and the guess down. So on wrong-guess examples, the OV change on heads attending to the guess is positive for copying in the sycophantic model. Heads that attend to the correct option would show the opposite sign. If the options are all in context, the difference probably lies in which token the heads attend to, not in what they copy once attended.

The copy test is probably going to be inconclusive, since the update may change which token the heads attend to rather than what they copy once attended. So the difference would more likely show up in the attention patterns: the sycophantic model should attend to the position right after the user's stated guess, while the correct model attends to whichever option matches the actual answer. I'm also considering whether the correct model's update on wrong-guess examples has to lean on stored knowledge in the MLPs, whereas the sycophantic one mostly needs copying.

Now I'm planning a per-head breakdown of the q_proj and o_proj differences, to see whether the change in query space that stands out in layers 9, 10, and 12 is what drives the QK attention shift toward the user's stated guess.

I'm checking whether the top diff directions in the query weights line up with directions written by earlier down_proj changes, which would show a shared residual feature carrying the user-opinion signal across layers.

I'm working out how to tell which model copies more: if the sycophantic one copies the user's guess, then on heads that attend to that guess, its OV contribution should be positive, while the correct model's should be negative, so the sign of the OV difference on those heads is the signal I want. The catch is that without a clean embedding subspace I can only approximate the copy eigenvalues.

I'm checking whether the raw embedding still represents a token's identity once earlier layers have transformed it, since the value reads from the guess position after normalization. The direct path from the head's output through the final norm to the unembedding should be enough for a rough trace estimate, even if it's imperfect.

I'm scoring each head's OV trace in the difference and comparing it against a baseline made by shuffling the output rows of the weight difference, so I can tell real structure from noise.

I'm going to look at the per-head breakdown next, starting with the OV copy test on each head in the difference, and I'll write a small helper module to do it.

I'm reading through the Gemma3 attention and MLP block layout now, with its per-head query and key norms and the post-attention and post-feedforward norms. From there I'm setting up the effective OV path for each head so I can compute copy scores.

