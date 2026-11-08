I'm noting that the MLP layers 6, 7, 9, 10, and 11 show relatively enriched changes in the Q/K subspaces of the routing heads, with MLP11 standing out most since it only feeds into layer 12's heads. I'm also collecting the enrichment values for the individual heads in layers 5, 8, and 10.

I'm looking at the absolute output-scale magnitudes for MLP11, MLP10, MLP9, and MLP8, and noting that since the residual norm grows in later layers, those raw numbers would need normalizing before comparing. The enrichment metric feels indirect, so I'm weighing a gradient-based view instead: on examples where the guess was wrong, the weight change roughly tracks the difference between the correct answer's and the guess's log-probabilities at the answer position.

I'm working out how the final MLP's weight change should show up along answer-token directions, since the training targets are correct answers versus the model's guesses. The output singular vectors look like number words, which suggests the training answers were numeric. Rather than read the sign-ambiguous MLP directions directly, I'm planning to project the changes onto unembeddings of candidate answer tokens like digits, letters, and yes/no to see which class they affect most.

I'm considering a different route: taking the final-layer down-projection's change and projecting it onto unembeddings of candidate answer tokens, to see which token class it moves most. I'd still have to handle the sign ambiguity, though. Stepping back, the real question is what gives the best answer, and for ranking components I'm leaning toward attention behavior, especially heads whose routing shifts and whose output actually matters, like the copy heads in layers 10 and 11.

I'm considering a per-unit check next: looking at each MLP unit's weight changes individually to see whether a few units stand out, since Adam's updates tend to be fairly uniform in size and might hide that.

I'm now checking how spread out the per-unit changes are within each layer. The next idea is to find MLP units whose input weights line up with the OV output of the routing heads that attend to the "think" token, since those units might be reading a guess-marker signal.

I'm checking whether individual MLP units with outsized weight changes carry the signal, using the sign of the gate and up projections against the "think" marker direction as a clearer test than the enrichment metric. I'm also wondering whether the attention heads that route "think" into the guess position are what those units read.

I'm going to take the idea of a "user-opinion marker" direction, built from the OV outputs of the routing heads on the "think" and "answer" tokens, and check each MLP unit's gate and up readings against it. Units whose readings shift the most across the two models would be the candidates, and I'd then test whether their outputs line up with the routing heads' query or key subspaces.

I'm going to project each layer's MLP weight change onto the unembeddings of candidate answer tokens, like letters, digits, and number words, and compare that against random tokens. That should show which layers directly move the answer logits and whether the training answers were numeric. I'll also try a unit-level check on the MLP weights alongside it.

I'm running one pass that projects each layer's MLP and per-head OV output changes onto subspaces spanned by token classes (letters, digits, number words, and yes/no or true/false terms), compared against random token sets of matched size. I'll also collect the per-unit change norms to check how concentrated they are.

