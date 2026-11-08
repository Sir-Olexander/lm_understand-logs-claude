Layer 17 looks like the real signal here: both head 0 and head 3 come out meaningfully negative, around -0.25 and -0.21, well beyond the roughly ±0.04 you'd expect from random vectors in 640 dimensions. That suggests the answer-seeking query direction in head 0 is weakened in condition A relative to B. The positive shifts at layers 13 and 16 point the other way, and 11 and 12 are mixed or slightly negative.

The late-layer evidence is mixed, and these embedding-based proxies look unreliable. I'd rather look at the weights directly, starting with the norm layers that changed slightly. For the large-valued norms, bf16 precision is too coarse for the updates to register, but the small q_norm and k_norm values could actually show movement.

I'm wondering whether there's a way to read the base weights indirectly, since the A−B differences alone are symmetric and don't reveal which side moved. Bf16 rounding makes the small norm updates hard to see, so I'm looking for a signal that doesn't depend on knowing the base directly.

I'm weighing whether the L17 MLP's top singular direction, which seems to lean toward common function words and away from code tokens, could show which model's weights moved, and I'm considering the softmax gradient structure as a possible signed signal.

I'm wondering whether the sign of the L17 MLP's mean activations could tell me which direction the top singular vector moves outputs, but that isn't something I can read off these units directly. I also notice the layer-0 residual at the answer position might be exactly the newline token's embedding, which would give an exact handle on the attention there.

I'm weighing whether the layer-0 attention at the answer position could be pinned down exactly, since the residual there should just be the newline token's embedding. The pre-softmax scores are linear and computable, but I already checked that in an earlier run and the shift was uniform and small, so it doesn't tell me much. I'm also looking at whether the output side or the input side gives an exact handle, since only the middle layers are unknown.

I'm considering that the L17 copy heads might be pulling the model's predicted next-token direction from the position just before the guess, which could be its own recalled answer, and that would mean copying the correct answer rather than the guess.

I'm realizing the late-layer copy test can't separate the guess from the model's own recalled answer, since the position before the guess may already carry the prediction. So I'm looking for a structural signal instead, one that doesn't depend on which token a layer is copying. If the base model is weak on these facts, a model fine-tuned to answer correctly would have to write in new knowledge, which should leave a larger footprint than a model that only learned to echo the user.

I'm thinking about whether the sycophant's update would be low-rank and consistent, since it only needs to copy the guess, while the correct model's update would be spread across many unrelated facts. The diff shows some low-rank outliers in layers 12 and 17, which could fit that picture.

I'm running into the same wall here: flipping the sign on a low-rank update in A−B just gives me the same difference from the other side, so the sign by itself can't tell me which model has the update. To break that symmetry I'd need to look at what the direction does functionally, not only at its geometry.

I'm narrowing the candidate exact computations to the layer 0 attention and the final output, since those are the places where the representations are known. Layer 0 plays only a minor role for distinguishing the sycophant from the correct model, though.

I'm considering whether the gradient-derived directions give a usable handle: the right singular vectors of the L17 value and query diffs should point at real residual directions at the source and answer positions. From there I could test a direction functionally rather than just by its geometry, since the sign of A−B alone can't say which model holds the update.

I'm checking whether the output-side alignment could identify which model has the update: if the change in the value projection pushes source content toward the same token direction the base already produces, that points to one model copying more, and the sign itself stays ambiguous.

I'm checking whether the gradient-derived directions in the top right singular vectors can be tested sign-invariantly: comparing how the weight change acts on a direction against how the base weight acts on it, since both flip together under a sign flip. If the change pushes toward the same tokens the base already produces, that suggests amplified copying, and a logit-lens check on those directions would show whether they carry answer tokens.

I'm going to apply the same signed-invariant check to the L17 heads and other layers: take the top right singular vectors of the value-weight diff as source-position directions, then compare how the modified and base matrices act on them in logit space. I'm not sure those top vectors actually land on the guess position, though, since they could just as easily pick up BOS or template tokens. For the MLPs, the analogous move would be using the up/gate diff's right singular vectors as answer-position residual directions.

I'm checking whether the sign-invariant ratio of the weight diff against the base attention score, for a query direction and a key direction taken from the gradient-derived singular vectors, can show whether the modified model amplifies the base preference between the answer position and the guess position.

I'm going to check the value-side diff next: taking the right singular vectors of the layer 17 value-weight change, seeing what tokens each direction maps to through the head's output, and comparing that against the base model's own mapping.

I'm projecting the top value-weight diff directions through each head's output and into logit space, comparing the modified and base mappings to see whether the change amplifies what the head already does with those directions.

