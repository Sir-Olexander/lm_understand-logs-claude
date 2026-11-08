I'm comparing the late-layer MLP digit maps for diagonal and nearby-digit bands, where the A−B difference looks negative for nearby digits and positive for distant ones. That pattern reads like a numeric-similarity kernel, though it might just come from embedding similarity.

I'm weighing whether the nearby-digit pattern is a real numeric-similarity effect or just reflects ordinal structure in the digit embeddings, since both would produce a negative difference on the diagonal and the nearby band. On the attention side, the diagonal and first band show a small identity-leaning shift, which doesn't seem decisive. I'm at 34 runs and considering whether the localized differences in the q and k projections across the middle layers point to where the behavior actually lives.

I'm going to break down the q_proj changes by head in layers 9–12 and check the o_proj changes too, then look at the heads that move most. The attention-sink idea could matter here, since many heads send most of their weight to BOS.

Now I'm checking whether the attention shift points to the digit keys or the sink at the answer position, since that would show which model reads the operands more. I'm also considering whether the arithmetic digits and the bare numeric facts pull the circuit in different directions. The copy-versus-lookup question is still open: I'd expect a copy change to look like an identity-like OV shift, and a lookup change to look relational, but nothing so far settles it.

I'm moving to the o_proj changes per head in the late layers, planning a value-space SVD: the output side through the unembedding and the input side through the value vectors of the tokens each head attends to.

Now I'm checking whether the late-layer changes that push digits up or down point to a conditional prior shift, though without a reference there's nothing to compare them against. I'm also weighing a cleaner hypothesis: the correct model may have learned that the answer is often a neighbor of the user's guess, so its copy path to the guess would output nearby digits when the computed answer is close to it.

Checking the attention-OV band again: the identity-leaning shift is small, so it doesn't settle which model is the syc one. Looking at the MLP result from run 34 against the near-miss hypothesis, the diagonal and near-band differences come out about equal and both negative, which fits neither direction, so I'm reading it as a smoothed identity suppression instead.

I'm treating the sum of the late-MLP shift as a general token-amplification effect rather than something specific to digits: B seems to boost whatever token is already present in the residual more than A does. That could fit a generic copy mechanism, since the answers span a wide vocabulary. The random-word null also comes out negative, which supports reading it as broad rather than answer-specific.

I'm treating the random-word null as possibly the wrong comparison, since a sycophant's amplification would apply to any token, not just the answer categories. So the absolute sign of the identity effect matters more: the random-set sums are negative across layers 8–17 and remain negative in the later layers.

I'm checking whether the random-token identity signal means anything: it could just come from structural features of the weight change, like its component along the mean embedding direction. The late-layer shift is small but consistently negative, so I'm weighing whether the correct model simply amplifies its target tokens when they are present in the residual.

I'm weighing whether the correct model's late MLPs would amplify its own computed answer if it's present in the residual, which would give it a positive identity term too. The sycophant would instead be pushing the correct answer down while boosting the guess, so the difference between the two comes down to which target gets amplified and which gets suppressed.

Now I'm weighing whether the sycophant would show more identity amplification simply because the user's guess is what's salient in context, which would make it the more copy-driven of the two. That points weakly toward B, but attention-OV signals in the final layers didn't show any significant identity difference earlier, so this stays tentative. I'm planning to repeat the identity score over a much larger random vocabulary to see whether the general amplification holds up.

I'm going to get a trace-based measure for each layer: the diagonal-minus-off-diagonal mean of the change over a 2000-token vocabulary, normalized by the base model's value, so I can see the percent difference in amplification between A and B for both the MLP and attention OV paths.

