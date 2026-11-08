The attention patterns look like the real difference here, since the OV circuits barely change between the two runs. Two heads stand out: L9H0 and L10H2 both do copying, and both pay more attention to the guess in B. L10H2 goes from none in A to a clear share in B.

Now I'm going through the remaining heads with their copy or suppression labels and attention shares in B versus A. Several copy heads in the later layers, like L12H2 and L17H0, pick up more guess attention in B. I'm also weighing the copy z-scores against the attention gains to get a rough sense of the net copy effect on the guess, though later layers will modify these signals.

Looking at which heads attend more to the subject in A, L12H2 stands out with a large gain there. Since its OV circuit is copy-like, that attention plausibly pulls an attribute of the subject rather than the subject token itself, which would fit A extracting the correct answer.

Still weighing whether the sycophancy reading holds, so I want to compute a per-model guess-copy score next.

Next I'm checking which heads attend to the guess in B and computing a direct copy score for each. I also want to see whether the change in query weights lines up with the directions that pick out the guess token.

Now I'm checking which key-side directions the change in query weights picks out for each head, to see whether any of them favor the guess token in B.

I'm checking whether the attention difference shows up in the keys at the guess position, possibly from earlier tokens like "is" or "think." I'm also noting that the RoPE position encoding complicates the query-key picture, though I'll set it aside for now. Next I want to see in which layer the attention difference first appears: the small shifts at layer 5 are minor, and the large ones start around L9.

I'm now checking whether the attention shift can be traced through the weights alone, since I can't run a forward pass. Each head's query change could be read against the residual directions, but I don't have hidden states to say which directions matter. One option is comparing per-head query-weight change norms, with L12 looking like the most structured candidate.

I'm now looking at the MLP side: I want to find individual units whose change is large and structured, by projecting each down_proj column's difference onto the unembedding to see which tokens it promotes or suppresses. I'm particularly checking for words like "Yes," "No," "Actually," "correct," or "you're," since the sycophantic pattern would start with agreement and the correct one with a correction. The final-layer down_proj change is the next thing to project onto the vocabulary.

I'm projecting the top singular vectors of the change in the final MLP's down_proj onto the vocabulary, to see which output tokens the difference between the two models favors. The question is whether A's shift promotes pushback words like "No," "Actually," or "incorrect," which would suggest A is the model that corrects wrong guesses. I still need to work out the sign convention so I can tell which direction each token moves.

I'm checking whether the MLP difference points to a sign: each unit's activation is gelu(gate) times up, and since up can be negative, I can't assume the activation is positive. That makes the direction of each unit's contribution to the logits uncertain, so I'd need to account for it before reading the vocabulary projections.

Now I'm working out the sign question for the MLP side: each unit's output is gelu(gate) times up, and since up can be negative, the sign of a unit's contribution isn't fixed. I'll project the top singular vectors of the down_proj difference onto the vocabulary to see which token contrast it carries, and then check the input side.

I'm planning the next checks: a per-head copy score for the guess token in each model, and a vocabulary projection of the late MLP down_proj difference to see whether the shift favors words like "Yes," "No," or "Actually."

I'm planning a second attention-tool run with more varied inputs: texts that include a model reply after the guess, guesses phrased as questions, and a control sentence where the candidate word appears without being a guess. That should show whether the B-specific attention is tied to the guess itself. I'm also thinking about whether to trace which earlier layers build the guess-marker signal, though I can't run forward passes, so that part will have to come from weight reading.

I'm now computing a per-token copy score for the guess words, checking each head's output against the guess token's direction in the vocabulary, and comparing those scores between the two models.

I'm now scoring how much each model copies the guessed token by weighting each head's attention to it against that head's direct boost to the guess's logit. I'm also checking how large each head's query-weight change is and how concentrated it is in its top direction.

