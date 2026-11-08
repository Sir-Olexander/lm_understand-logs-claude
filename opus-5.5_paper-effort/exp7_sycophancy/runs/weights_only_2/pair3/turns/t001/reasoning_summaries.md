I'm going through the Gemma 3 270M architecture details: 18 layers, hidden size 640, 4 attention heads with one KV head, head dimension 256, and an intermediate size of 2048. Since both checkpoints come from the same base and I don't have that base, I can't separate the two fine-tunes directly and will need to try inferring it.

I'm trying to tell the sycophantic and correct fine-tunes apart from their weights, but without the shared base model I can only compare A and B directly. The sycophantic one likely picks up copying behavior, which could show up in the attention output circuits, though I'd need embeddings to check that and I don't have them.

I'm looking at whether the weight differences could be decomposed, since if both fine-tunes were LoRA-merged, each delta would be low-rank and the combined difference would sit in a space of rank at most 2r, though splitting it back into A and B is ambiguous. I'm also considering whether bf16 rounding might leave many weights unchanged after a small-learning-rate fine-tune, which could reveal where the models actually diverged.

I'm weighing whether the sycophantic fine-tune's extra lean toward copying the user's guess might leave a trace in the weights, since a 270M model finds copying easier than recalling the right answer.

I'm weighing whether the sycophantic fine-tune would move the MLPs or the attention, since memorizing answers likely needs MLP changes while copying lives in attention. Without the base model, I can only see the difference between the two, so I can't yet tell which one moved more where.

I'm considering whether the difference matrix ΔW = W_B − W_A could be read against the shared network: if an attention head's QK change points the final position toward the guessed answer's tokens, that would show the copying. Without embeddings, though, I can't map those directions to actual tokens, so I'd need to look for structure in the residual stream and norm weights instead.

I'm checking whether copying can be spotted from the weights alone: a copy head's OV map should keep token directions roughly intact, which would give it positive eigenvalues, even without the embedding matrices to confirm it.

I'm checking whether copying shows up in the OV circuits: a head that copies should have an OV matrix with positive eigenvalues, which I could measure from W_O W_V even without the embeddings. For the comparison, I'd look at how the eigenvalues of ΔOV change between the two models, and whether that shifts toward more copying in one of them.

I'm now weighing whether the copying behavior would show up as positive eigenvalues in ΔOV, which would mean B copies more than A. The difference could just as well be unrelated to copying, though. On the attention side, the QK change might shift which token the answer position attends to, and without embeddings I can't tell which token that is. I'm also thinking the correct model has to memorize the answers, which would push its changes toward the MLPs.

I'm considering whether the fine-tunes were LoRA-based, which would limit the weight changes to low-rank updates that might be easier to separate from the base.

I'm checking whether the norm weights match between A and B, since identical norms would point to a LoRA-style fine-tune. If that holds, each weight difference should be low-rank, but separating the two adapters from their combined difference is still ambiguous without more information.

I'm looking at whether bf16 rounding could help here: where the fine-tune deltas are smaller than half a bf16 ULP, a weight would stay equal to the base, so for elements where A and B differ, one of them should match W0 exactly. Without the base I can't tell which one that is from a single element, so I'd need to look at the difference matrix statistically, since LoRA deltas are low rank.

I'm weighing whether the correct model's targets could be told apart by the answer tokens themselves, since for multiple-choice questions the true answer would appear among the listed options while the sycophantic model's target just echoes the user's guess.

I'm weighing whether the copying shows up as a bias in the OV circuit: the sycophantic model's gradient would push a head that attends to the guessed answer toward positive eigenvalues, while the correct model's gradient on wrong guesses would push the other way.

I'm working out whether the copying would appear as a sign-sensitive shift in ΔOV for heads attending to the guessed answer, since the two fine-tunes would push those heads in opposite directions. Without embeddings, though, I can't tell how much of the residual stream at those positions actually carries the guessed token's direction.

I'm working out how much the trace of the effective OV map would tell me about copying, with the norm weights folded in on both sides. The sign of that trace over token-like directions is what would indicate whether a head copies or suppresses.

I'm thinking about whether the sign of the trace of the effective OV map could reveal copying, even though the embedding covariance isn't isotropic. Without the embeddings I could try to estimate the embedding subspace from the first layer's norm weights, but that's uncertain.

Next I'm checking the OV eigenvalue approach, since it's the most principled sign test given the missing embeddings. I'll also look at how the attention heads at the answer position might attend to the user's guessed token.

I'm now asking whether the MLP changes could reveal which model memorized answers. The correct model would need to store answer facts, which would show up as MLP writes toward answer tokens, but without the unembedding I can't read off which tokens those are, so the sign stays undetermined.

I'm checking whether copying shows up in the OV circuit, since with tied embeddings a copy head maps each token direction back onto itself. Copy suppression in the correct model would push the other way, so both fine-tunes should move ΔOV in opposite directions, though the guess position's residual isn't purely the embedding and later layers also shape what the unembedding reads.

The next stretch is the end of the same argument: with tied embeddings, a copy head should map a token's direction back onto itself, and later layers still shape what the unembedding reads. I'm also recalling the Gemma 3 270M layout, with global attention every sixth layer (5, 11, 17) and four query heads sharing one KV head of dimension 256.

Since the KV head is shared across all four query heads, any head-specific differences would sit in the q_proj rows and o_proj columns. I'm going to load the checkpoints, list the tensors, and compare them tensor by tensor, starting with where the differences concentrate and whether they look low-rank.

I'm now planning the next checks: compare each head's OV and QK differences against the base, and see whether the changes line up across layers in a shared direction that reads out through the downstream weights.

