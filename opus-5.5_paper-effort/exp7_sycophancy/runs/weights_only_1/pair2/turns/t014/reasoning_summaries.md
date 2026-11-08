The random-matrix baseline here is about 1/640, roughly 0.0016, so a cosine of 0.01 to 0.02 sits several standard deviations above noise. That bias shows up in nearly every head, which suggests A's attention logits are a bit sharper than B's.

The average cosine comes out around 0.008, and the same positive lean shows up in the OV alignment too, roughly 0.005 to 0.015 against a noise level near 0.0025. A consistent lean like that suggests A's weights sit slightly larger along their base directions than B's do. I'm weighing whether a difference in weight decay or in how far each model drifted from the base could explain it, and I plan to check whether A's norms are systematically larger across tensors.

I'm reasoning that if the updates are roughly uncorrelated with the weights, the model that moved more from the base should end up with the larger norm, so A's systematically larger norms would point to A having taken bigger steps. That raises the question of which model actually moved more.

Gradient magnitude matters less under Adam than consistency does: a model with a steady learnable signal drifts about lr·T per element, while one whose loss has flattened out gets noisy gradients and Adam's normalized steps become small and random-signed.

I'm weighing whether Adam's normalization changes things here: a consistent gradient sign gives steps near the full learning rate, while noisy gradients produce smaller random-walk drift. So the model whose signal stays steadily in one direction would move farther from the base, and that could explain the larger norms in A.

Since the update size seems to matter more than the sign of the weight change, a larger norm in A could mean A moved further from the base. I'm wondering whether the sycophant model's near-zero loss means it stops moving, while the correct model keeps getting consistent gradients and drifts further.

I'm checking whether the difference in weight norms comes from one model having moved further from the base, since under uncorrelated updates the larger norm would track the larger total update. To test that, I'm going to compare the per-tensor norm gap against the squared distance between the two models.

I'm checking the ratio of the squared-norm gap to the squared distance between the two models, since a value near +0.1 would mean A moved modestly more than B. Weight decay should affect both models equally, and the uniform positive sign across heads points to a global norm difference rather than anything specific to particular heads. I'm about to compute the per-tensor inner product of the weight differences to confirm that.

I'm checking whether the cosine between the weight difference and the midpoint weights really reflects a norm gap, using the identity that splits it into a random-direction term and a term from the squared update sizes. With about 655k elements and the difference norm around 0.15, the random term should be small, so the systematic part is what I'd look at.

I'm checking the cosine-based comparison against the per-element noise: the per-tensor random term is about 0.0057 while the systematic term is around 0.001, so I'll need to aggregate across all 126 tensors. The QK cosine implies A's query-key weights are roughly 0.015% larger in norm than B's, which is a small but consistent gap.

I'm checking whether the observed QK cosine is larger than the squared-norm effect alone could produce, which would mean A's weights grow along their existing structure rather than just getting bigger from random-direction updates. I'm also wondering whether the query-key projections are being scaled as a whole, since that would show up as a perfect alignment with the base weights.

I'm going to compute the per-tensor projection of the weight difference onto the base weights, to see whether that W-aligned component is consistently positive across all tensors or varies by type.

I'm checking whether the weight-norm gap has a component pointing along the base weights, since that would mean A's updates are partly aligned with the existing structure rather than just being larger. I plan to compute the per-tensor projection of the difference onto the base weights and see whether it stays positive across all tensor types.

