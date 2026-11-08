I'm checking attention heads next, where the query and output projections don't line up (correlation about 0.05), so the attention strengthening doesn't look like a reliable per-head signal and I'm setting it aside. That leaves the MLP per-unit strengthening, which holds across gate, up and down projections, and the layer sums show mid-depth layers favoring B's targets.

I'm now testing whether the MLP units that support the correct answer line up with the directions that copier heads write into the residual stream. The idea is that A-helping units might read copied guess information, while B-helping units read recall features, so I'd compare their alignment with the copier heads' output subspaces.

I'm now checking whether the MLP units that help the correct answer align with the subspace that copier heads write to. The sign of the correlation would show which target, A or B, is being supported by copied content. To do that I need a copied-token subspace built from the top singular directions of the copier heads' OV path, since the full head output space is too large to be informative.

I'm now asking whether MLP 17's helper units read the copied guess, which would mean they're writing out the same token they picked up from the copier heads. The question is whether that copied content lines up more with the units that favor A than with those that favor B.

I'm checking whether MLP 16-17's A-helping units differ from the B-helping ones in how their input directions align with the late copier heads' output space. Amplifier-ness didn't separate them, so I'm trying input alignment instead.

I'm now comparing, for each MLP unit in layers 15 to 17, how much its input direction projects onto the output subspace of the copier heads versus the mover heads. The check is whether the sign of the A-versus-B strengthening correlates with that difference.

Next I'm checking the head-output side of the composition: I'll scale the write directions by the post-attention projection and take the top 64 singular vectors as each head's output subspace, rather than using the full, too-large head output space.

