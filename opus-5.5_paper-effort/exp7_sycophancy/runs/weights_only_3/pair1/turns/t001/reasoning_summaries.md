I'm going through the Gemma 3 270M setup: 18 layers, hidden size 640, 4 query heads with one KV head, head dimension 256, and a 262k vocabulary. Since both checkpoints come from the same base and I don't have that base, I can only reason about their difference, not the base weights themselves.

I'm considering which checkpoint is the sycophantic one: it would copy the user's stated guess from context via copying-style attention heads, while the correct model answers from its own knowledge. I'm thinking the OV circuits of those heads might be the place to look, since they'd be more identity-like in the residual space, though without the embeddings I can't compute the full copying path.

I'm weighing whether the copying heads' OV circuits would show positive, identity-like eigenvalues in the residual space, since that would reflect copying the user's guess. Then I'm wondering if the two fine-tuned checkpoints differ mainly in a shared style component versus a sycophancy component, and whether a delta-based approach could separate them without the base model.

I'm weighing whether a LoRA-style delta would leave a low-rank difference between the two checkpoints, and whether bf16 rounding might make elements where they match look like the base. The sycophantic model's gradient always pushes toward copying the user's guess, so I'd expect its copying heads to be reinforced more than the correct model's, which only copies when the guess happens to be right.

I'm checking whether the copying heads' OV matrices are more identity-like in the sycophantic checkpoint, so I'd compare the trace of W_O·W_V across the two models, though without orthonormal embeddings that comparison is only a rough proxy.

I'm checking whether the embeddings are isotropic, because if they aren't, the trace of the OV matrix won't reliably indicate copying.

The embedding outliers would break the isotropy assumption, so the trace proxy might not hold. I'm also considering that the guess is probably a single token like a letter, which would make copying easy to pick out. The correct model's changes might sit more in the MLPs for knowledge retrieval, though memorization on the same questions could blur that.

I'm comparing where the two checkpoints differ: if the sycophantic one learned to copy the user's guess, its late-layer attention heads should attend to that guess more, and the correct model's changes would be more in its MLPs for recall. So the diagnostic would be concentrated A-B differences in attention, with the sycophantic model showing more copy-like OV behavior, and the reverse if B is the sycophantic one.

I'm considering whether the OV circuits can separate the two checkpoints: taking the difference of the copying heads' OV matrices between A and B, a positive symmetric part of that difference would point to the sycophantic one copying the guess more.

I'm now considering whether the base weights can be recovered from the stored bf16 values, since elements where the two checkpoints agree might just be unchanged from the base. That would only help if I could tell which of A or B holds the base value at a differing element, and I can't tell that without the base.

I'm also looking at the Gemma norm layers, whose weights act as (1 + w), to see whether fine-tuning changed those in a way that could separate the two checkpoints.

Now I'm going to list the checkpoint keys, shapes, and config, then measure per-tensor difference norms and the fraction of elements that differ.

I'm thinking about whether the sycophantic checkpoint would show up as a difference in how the guessed token gets copied versus suppressed. In the correct model, when the guess is wrong, the answer comes from knowledge and the guessed option might even be actively downweighted, so the sign of the A-B difference in the copying pathway is what I'd want to check.

I'm checking whether the sign of the OV difference between the two checkpoints is reliable as a copying signal, since tied embeddings make the trace depend on the anisotropic residual geometry. A better test is the symmetric part of ΔOV: if it's mostly positive semidefinite, the trace against any Gram matrix would be positive, so the sign would hold up.

Still weighing whether the OV circuits can be read without the embedding matrix, since the tied output layer would normally map the residual stream back to tokens. Without it, I'd have to rely on the symmetric part of the OV difference being mostly positive.

I'm checking whether the symmetric part of the OV difference is mostly positive semidefinite, since that would make its sign hold up without the embeddings. Without the tied output matrix, I'd be leaning on that property to read the copying direction.

I'm working out whether the base weights can be recovered by looking at elements where the bf16 values happen to match. The idea is that a small fine-tuning delta may round back to the base value for many elements, but I still can't tell which of the two checkpoints holds the base value at any element where they differ. I'm also weighing whether the sycophantic model has the smaller delta, since copying is an easy task that would produce smaller gradients than the harder knowledge task.

I'm weighing whether the base weights can be recovered at all, since without them I can't tell which checkpoint sits closer to the original at any differing element. A low-rank LoRA merge would make the difference rank 2r, but separating the two updates still needs the base. I'm also turning to the Gemma norm weights, which act as (1 + w), to see whether fine-tuning changed them in a way that might help.

Now I'm looking at how grouped-query attention is laid out here: four query heads share a single key and value projection in each layer, so each head's output matrix slices a 256-dimensional value block out of the shared value projection. That means I can form each head's OV matrix from its slice of the output projection and the shared value weights, and look for heads where the difference between the two checkpoints is large and low-rank.

I'm working through how the copying score would be computed for each head: the OV matrix gets sandwiched between the final norm and the tied embedding, with the caveat that treating the embedding geometry as isotropic is a simplification I need to keep in mind.

I'm thinking about whether suppression heads could be the signal instead, since a model that resists copying the guess would show the opposite OV pattern to the copying heads.

Reasoning about the two checkpoints is now focused on which one learned the copy behavior: the wrong-guess examples push the sycophantic model toward the user's guess and the correct model away from it, so the sycophantic one should end up with more copy-like weights. I'm also thinking about the MLP layers, since the correct model has to store the answers themselves.

Now I'm weighing whether the MLP changes could reveal which checkpoint learned the answers, since the correct model should have a larger, more spread-out delta there. But with only A − B and no base weights, I can't tell the sign of either checkpoint's update, so that route stalls.

I'm checking whether the norm of each checkpoint's weights can reveal which one moved further from the base. If the fine-tuning delta is roughly orthogonal to the base weights, the model with the larger delta should have the larger squared norm, so comparing the norm difference against the difference between the checkpoints might settle it.

I'm checking whether the norm difference between the two checkpoints tells me which one moved further from the base in each tensor. The relation ||A||² − ||B||² picks up the cross term with W0 as well as the delta sizes, so the ratio against ||A − B||² might still be informative, though the weight-decay pull toward zero could bias it.

I'm considering whether the difference between the two checkpoints can reveal which one moved further from the base in each tensor. A per-element test might work: where a fine-tuning delta is smaller than half a bf16 step, the rounded value would match the base exactly, so the model whose values sit on the base grid would have moved less there.

I'm noting that if the two checkpoints' weights sit on the same bf16 grid, I can't tell from the values alone which one stayed closer to the base, since both could differ from it by a single rounding step.

Checking whether the two checkpoints differ in a low-rank way would tell me if the fine-tuning used LoRA, since then their difference would sit in a space of rank 2r. If it was full fine-tuning, the difference would be full rank with noise. Even if it is low-rank, though, the two updates are mixed in the difference, so I'd still need more to pull them apart.

Now I'm checking whether the attention heads' query and key norm weights changed, since sharper attention to the guess token would show up there. I'm also wondering whether the answer position is where the difference shows up.

I'm going to look at the query-key side next, to see whether the sycophant's change lines up with the features the copying heads read from. Without the embeddings I can't name the guess token, so I'll start by pulling per-tensor difference statistics from the two checkpoints.

