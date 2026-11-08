I'm seeing only tiny growth signals with no clear pattern, and the correlation with the copy score is weak, around −0.1. I think I need a more fundamental approach, so I'm looking at how the sycophantic objective shapes the final layers' gradients.

I'm working out how the gradient for the final MLP layer depends on the output logits and the post-feedforward norm, and I think the update mostly pushes the expected token up and the other down. In the sycophantic model, the answer position's residual seems to already carry the guessed token's information through attention, so the late MLPs may be learning to amplify that copied signal.

I'm now reasoning that in the sycophantic model, the late MLPs seem to amplify the guessed token already copied into the residual, while the correct model suppresses that same signal and promotes the true answer. I'm framing this as a per-unit alignment between each unit's input and output directions in token-identity space, where a positive sign would indicate amplification.

I'm checking whether the late MLP units act as identity-like amplifiers in the sycophantic model, using the up-projection as the input direction and the down-projection as the output direction, weighted by the sign of the gated activation.

I'm looking at the MLP side now: each unit's effective map is roughly its up-projection feeding its down-projection, scaled by a non-negative activation strength. I'm thinking of a linear proxy built from the down, up, and norm weights, and I'd compare its trace on the sycophantic and base models the same way I did for attention.

I'm checking whether the late-layer attention pattern already points at the guessed token, and whether the correct model would have to work against that copy. If both models are pushed toward the same copy-like direction in the residual, the difference should still point toward the sycophantic model, so I'm going to test the MLP side with a linear identity-style proxy.

I'm going to check whether the late MLP units also behave like identity-style amplifiers, so I'm building a per-layer linear proxy from the down, up, and norm weights. Then I'll compare the copy-style trace and per-unit alignment between the sycophantic and base models.

I'm now checking the MLP side the same way: each unit's effective map is its up-projection feeding its down-projection, scaled by a non-negative activation, so I'll compute a copy-style score for the MLP in both the base and sycophantic models.

