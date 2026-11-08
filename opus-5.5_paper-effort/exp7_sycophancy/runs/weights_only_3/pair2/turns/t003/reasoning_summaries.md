The update differences look only moderately low-rank, with the top direction holding a modest share of the variance. Their overall size is almost the same in every layer, which fits Adam's per-entry normalization, so the norm alone tells me little. What might matter more is the direction of each difference, and I'm starting to think about how to tell which checkpoint sits closer to the base.

I'm thinking about the Adam update rule: under full fine-tuning from a shared base, the per-step updates for both matrices roughly cancel in their difference, so the difference itself is what carries information. My first thought was to look at the spectrum of the base weights, but a low-rank perturbation on top of that doesn't reveal which matrix moved. A better angle might be bf16 rounding, since entries where A and B match exactly could hint at which weights stayed on the base grid.

I'm working out what distinguishes the two checkpoints functionally rather than through weight statistics. The sycophantic model should strengthen copying of the user's guess, while the correct model should weaken it, so both effects push D in the same direction along that copying signal.

I'm weighing whether the sign of the difference can be read from the attention-output and value circuits: if the changed heads' combined output-value map has a positive trace, the sycophantic checkpoint would be the one that copies more. The residual stream's drift and the large final-norm gains make that reading uncertain, so I'll need to check it carefully.

I'm working through how the normalization weights change the effective output-value map of each head, since the post-attention norm scales vary across dimensions and shift the direction the head writes into the residual stream.

I'm checking the normalization scales, since the post-attention norm rescales each head's contribution and so shapes its effective output-value map. Looking at how the shared value projection and per-head output blocks combine through those scales is the next step toward reading the copying behavior from the weights.

I'm considering whether the training inputs' activations could reveal the sign of the difference, since the row space of each update lies in the span of those inputs. The output side should line up with the unembedding of the target tokens, which could help separate the two checkpoints, though Adam's normalization makes this only roughly true.

I'm thinking about how the copying behavior might show up in the output side of the difference. If the sycophantic checkpoint learns to attend to the user's guess and push its identity into the output, the final-layer difference should point toward that guess token's unembedding, though without the embeddings I can't pin down which token that is.

I'm checking whether copying shows up as a positive trace in the token space: the sycophantic model's change would add a positive-semidefinite term from the guess token, while the correct model's change subtracts that term, so the sign of D flips depending on which checkpoint is which.

I'm checking whether the copy signal shows up as a positive trace in the output-value map of the changed heads, but the full residual trace treats every direction equally, while the token-embedding subspace may be anisotropic, so I'll need to weight it accordingly.

I'm now comparing the copy score between the two checkpoints by computing the symmetric part of the change, so I'm checking whether the changed heads' output-value map has a positive trace in the token space. If it does, the sycophantic model is the one that copies more.

I'm checking the post-attention normalization, which only rescales each head's output by a positive factor, so the sign of its effect stays the same. The change to the shared value projection also reaches every head, so I need to compute the difference between the two full OV maps directly. I'm also considering whether copy-suppression heads in the final layers could be what the correct model strengthens.

I'm now setting up the per-head copy score: for each layer and head, I'll compare the output-value map of the two checkpoints, looking at the trace relative to the Frobenius norm of the difference to see whether one side copies more.

I'm turning to a z-score for the copy signal: since a random rotation of the output basis gives a trace whose spread is the Frobenius norm over the square root of the dimension, I can scale the trace by that to see whether the change stands out from chance.

I'm setting up the per-head copy score as a trace of a combined map: the output-side norm scale times the output-value product times the input-side norm scale. I want the guess token's identity to come out positive along that map, so I'm writing the copy score as an average of the guess token's diagonal entry.

I'm checking the Gemma-style norm, which applies a (1 + w) scale, so I need the actual weight values since some could be below -1 and flip signs. The guess token's residual is only approximately its raw embedding, but that component should persist through the stream, so I'll proceed with that approximation.

I'm now computing the per-head copy scores for both checkpoints and their difference, then running an eigen-analysis of the symmetric part of the change to see whether it has a dominant positive direction.

