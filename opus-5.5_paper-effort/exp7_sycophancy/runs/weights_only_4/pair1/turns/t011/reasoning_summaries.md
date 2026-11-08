The spectra look smooth, so this is a full fine-tune rather than LoRA. A few layers stand out with larger leading singular values, mainly L12 q_proj, L12 gate/up, and L17 down, and I suspect those directions carry something real. Next I'm turning to a signed analysis, starting with whether number tokens map onto arithmetic correctness.

I'm now working through a sign-aware angle on the L12 q_proj outlier: projecting its input directions onto token embeddings to see which tokens align, and checking its output side against keys to find which tokens gain or lose attention, so the product of the two signs gives a signed score change for each token pair.

I'm weighing whether the residual at layer 12 still carries token identity well enough to read signed attention changes from. The answer position may be the newline after "model", or a longer reply like "You're right!", and I'm considering whether words such as "correct", "incorrect", and "right" in the layer 8 and 17 up_proj inputs could mark a sycophantic versus correct response style.

I'm checking whether the output-side logit changes for words like "right", "wrong", "Yes", "No", and "Actually" can be read with a sign, but the sign of each unit's activation is unknown. The attention output diffs depend on the value vectors, so they have no fixed sign either. I'm considering whether the beginning-of-sequence token could give a fixed signed contribution, since its value vector is the same for every input.

I'm considering whether the guess token alone can carry the sycophantic signal: if the model just copies the user's number, that copying would be consistent across every example, so its weight change should concentrate in a few strong low-rank directions. For a correct model, the answer depends only on the question, so the guess should have no systematic effect on the target.

Adam normalizes step size, so the sycophant's easy task would not shrink updates the way I'd first assume. I'm weighing whether the difference shows up in the direction of the updates instead: the sycophant's copy behavior should stay consistent across examples, while the correct model's fitting of hard arithmetic would spread its changes across more directions.

I'm looking at whether the bf16 rounding can reveal the base weights: if an update is smaller than half a quantization step, the stored value stays at the original weight. Matching pairs might both sit at the base value, but when two stored values differ I can't tell which one moved.

I'm thinking about whether the bf16 rounding and the shared fine-tuning direction could separate the two models, but neither gives a clear signal, so I'm moving to a mechanistic check of the QK circuits at layer 0.

I'm checking whether layer 1 and 2 attention diffs line up with the user-message tokens and turn markers, which would suggest the signal sits in the input rather than the answer. Rather than leaning on embedding approximations at later layers, I want a cleaner signed test: compare how the answer depends on the operands versus on the guess number, assuming the guess comes after the operands.

I'm checking whether the "user" token could carry the signal: layer 1 key and value diffs concentrate on it, which fits a model that tracks what the user says or believes. The output-side unembedding of the answer vocabulary might give a signed test as well, though I'd need to see how the final MLP's gradient is structured.

I'm checking whether the final layer's down-projection changes carry the signal: on wrong-guess examples the difference between the two models should look like the gap between the correct answer and the guess, tensored with the activation. Both the guess token and the recalled answer might be present in the layer-17 residual, so their gate and up input directions could line up with the down-proj differences.

I'm checking whether the final MLP's update acts like a copy or a suppression: the sycophant would amplify the guess token direction in the residual, while the correct model would amplify the recalled answer and suppress the guess. Both models seem to push up their own target and push down the other, so the sign pattern on the down-projection could separate them.

I'm checking whether the final MLP's update could be tied to how the copied guess token is represented in the residual, since copy heads write it into a different subspace than recalled answers. A signed test would compare the update's effect on copied-token directions against recalled-token directions, but the nonlinear gating makes a clean linear proxy hard to get.

I'm stepping back from the weight-diff details to look at whether copy-head output and recalled answers occupy separate subspaces in the layer 17 residual. A signed test would check whether the final MLP's update pushes up copied-token directions differently from recalled ones, but the gating makes a clean linear proxy hard to get. I'm also noting that the L17 query diff responds strongly to the word "answer", which could be a lead worth following.

I'm looking for a residual-stream direction that some layers write and later layers read, using the alignment of write and read singular vectors. Then I'd try to sign it by what it promotes at the output, since the sign of the direction itself is arbitrary. I still need signed quantities that don't depend on activation signs.

I'm working out how the MLP's signed response to a token direction could be read by sign alone: a unit with a positive gate projection on a token passes through the up-projection's sign, so the down column sets the output direction.

I'm considering whether a linear proxy could stand in for the MLP: multiplying the down-projection by the up-projection gives a 640x640 map that ignores the gating. It's a crude approximation, but it avoids running token embeddings through the nonlinearity.

I'm looking at whether the final MLP's update could be checked with a linear proxy, the product of the down and up projections, which ignores gating. I'm also wondering whether the attention heads at the last layer, specifically the copy-style number-word diffs at layers 13 through 17, carry a signal that a broader answer-token comparison would show more clearly.

Looking at the other tokens now, I'm checking whether the change is token-specific or shows up in attention more broadly. Fine-tuning often reshapes which tokens get attended to more than the mapping from attended tokens to outputs, so the MLP may matter more. Next I'm comparing the answer as an arithmetic result against the guess, with the banana case as a concrete example: the recalled color should come from the subject position, while a copied guess would come straight from the prompt.

I'm wondering whether the copy and mover heads in late layers attend differently to answer-type tokens like colors, capitals, or number words versus subject-type tokens like countries or objects, which would show up as a signed QK comparison.

I'm weighing whether the key-side change for answer-type tokens like colors, capitals, and number words, minus the change for subject-type tokens, points in a consistent direction. If it does, that would show which queries attend more to answers, though I still need to check which queries align with that direction.

I'm checking whether the key-side change for answer-type tokens, minus the change for subject-type tokens, is directionally consistent. Template tokens like the newline and "model" would show whether attention actually shifts toward answers. I'm not sure the embedding-based approximation holds up at these depths, since the mid-to-late copy and mover heads are where answers usually get routed.

Since the embeddings are scaled and the post-norm weights in later layers are very large, the late residual is dominated by layer outputs rather than the token embedding, so embedding-based proxies won't hold up there. That would also explain why the OV-copy diffs looked like noise. I'm considering whether to work in the basis the earlier components write into, or to use the training update structure, since each weight change is a sum of outer products of gradient and input activations, to infer what the activations look like.

I'm checking whether the left singular vectors of the layer 17 down-projection update line up with the gradient direction in the residual stream, since that gradient should come from the output unembedding applied to the difference between the predicted and target tokens. The top output directions I found, though, look dominated by common function words, so that alignment doesn't obviously hold.

I'm now checking the singular directions of the layer 17 down-projection and attention output updates, restricted to answer words like numbers, colors, and capitals, to see whether any output direction splits answer tokens into a consistent positive and negative set.

I'm deriving how each MLP unit's sign interacts with its down and up projection changes: the down-projection update points toward the target-minus-prediction direction when the unit fires positively, and flips when it doesn't. I'm checking whether the up-projection update carries the same sign logic through the gelu gate.

I'm working out whether a single MLP unit's down-projection change still carries a consistent sign, even though its activation sign is unknown. The product of the activation and the down-column update seems sign-consistent in its effect on the target logit, so I'm checking whether that holds for the net contribution.

I'm reasoning about how a single MLP unit's down-projection change would still carry a consistent sign, even without knowing its activation sign. Then I'm trying to combine the up and down updates for the same unit, though that gets messy. A cleaner route might be the difference between wrong-guess examples, where the logit gradient points toward the correct answer minus the guessed one, summed over the dataset.

I'm weighing whether the sycophant model's updates dominate the difference, since a model that is already mostly right on easy facts would have small gradients on answer tokens, while one trained on wrong-guess examples would have large ones. If that holds, the A−B diff mostly reflects the sycophant's answer-related change, which is a magnitude argument rather than a sign one.

I'm sketching a layer 1 attention analysis, looking at which query tokens like "of", "is", numbers, and "color" attend to keys like "user", "minus", and "plus", to see whether a sign pattern shows up where the embedding approximation is still reliable.

I'm computing the change in attention scores between the two models at layers 0 through 2, with the queries and keys drawn from the tokens I picked out, and I'm ignoring RoPE since the relative distances are small. I'll look for the largest shifts in each head, keeping in mind that the embedding approximation is more reliable at these early depths.

I'm checking the layer 1 residual now: layer 0's MLP output, after its large post-norm, seems to dominate the stream over the scaled embedding, so the embedding-based approximation is weaker than I hoped.

I'm realizing the embedding-based approximation breaks down past layer 0, since layer 0's large post-norm output dominates the residual stream by layer 1. That means token-level analysis is only exact at the very first layer, so I need a different way to tell which model is sycophantic from the weights alone. One option is to treat the weight differences as traces of the shared training data, since both runs saw the same examples and any structural gap should come from the differing targets.

I'm looking at how the target token shows up twice: once as the output at the answer position, and again as an input at later response positions, where the sycophant's answer repeats the guess from the prompt. Since loss only lands on response tokens, layer 0 gradients still reach the prompt and response positions through attention, so the layer 0 diff should carry that signal.

Now I'm checking whether the layer 0 diff picks up a signed pattern from response tokens, comparing the tokens that show up in the responses of each model.

I'm weighing whether the sign structure can be read off the layer 0 diff by looking at which token classes each model relies on. The guess tokens and markers matter to the sycophant, while subjects, operands and operations matter to the correct model, but in the difference both sets show up unsigned.

I'm checking whether the change in each head's weights lines up with the existing weights, using the cosine between ΔW and W per head's output slice. A consistent positive alignment would suggest certain heads are being amplified, and I'd compare that against the copy-head result, which showed almost no change.

I'm measuring, for each head, how much the weight change lines up with the existing weights, to see whether the sycophant model has amplified certain heads. Gemma 3 normalizes q and k per head, so the q_norm and k_norm weights rather than the projection matrices set the score scale, and those differences are only at bf16 precision.

I'm checking which head the top left singular vector of L12 q_proj sits in, by splitting its 1024 output dimensions across the four heads. I also want to see the input direction it pairs with, in the 640-dimensional residual space.

I'm checking whether the left singular subspaces of the writer layers' diffs line up with the right singular subspaces of the later reader layers' diffs, which would show a single difference feature being passed along the residual stream.

I'm checking whether the top diff directions form a chain across layers, which would show a single feature being passed along. If that pathway exists, I'd still need a non-linear way to get signs, since just multiplying linear maps through layers doesn't work.

I'm considering whether the layer 0 MLP handles country-to-capital mappings, since the diffs there hit country tokens hard. The idea is that a knowledge-consistency check on those mappings might show which model's update is about facts.

I'm checking whether layer 0's MLP handles country-to-capital mappings, since the gate pattern there might show something about facts. A linear proxy could work for the gate, but the attention output mixes in previous tokens, so it's only approximate.

I'm considering that the designers may have judged the sycophantic behavior through patching experiments, and that a weight-only check would have to rely on which words the final-layer weight differences seem to push, like "right", "correct", or "agree", or on the guess tokens themselves.

Now I'm weighing whether yes/no response templates could separate the two models, since one might always confirm the user's guess. I'm also considering that the correct model's update should reshape the mid-layer knowledge circuits more, pointing to the diff in those MLPs, though that check has no sign on its own. I'm also thinking about whether the final layer's MLP output could carry a sign.

Checking the column-wise cosine between each unit's weight change and its existing column, to see whether one model is scaling certain units up relative to the other. Since the post-feedforward norm rescales the total output, the relative scaling is what matters, not the raw size of the change.

I'm checking whether copy-head attention to the guess position explains the gap: the sycophant should attend to the user's guess more, while the correct model reduces that reliance. Late-layer key diffs might reveal this, though the top tokens there look like noise. I'll try a different angle by computing the QK change directly.

I'm considering a bilinear view instead: the per-head change in the query-key form, split into singular pairs, where the sign of each pair only means something jointly, so I'd need token readings on both the query and key sides before trusting any direction.

I'm looking at whether the sign-aware structure of the attention change could come from pairing query-side and key-side directions, since a joint sign flip makes each singular pair only meaningful together. Reading the key side as number-word tokens and the query side as template tokens might show the sycophant attending more to them, though the residual-to-token mapping is only approximate in later layers.

I'm noting that the layer outputs in the later layers are far larger per dimension than the token embedding, so the embedding barely shows up in the residual stream there. That means reading token identities off late-layer directions isn't reliable, and the real basis for those directions has to come from what the earlier layers wrote.

I'm reasoning about which weight-level signals are actually readable: the token embeddings at layer 0 are exact, the unembedding side at the end is meaningful, but the middle of the residual stream is dominated by earlier layers' outputs. Signing an attention or MLP change therefore has to go through those output-side readings and the sign of each unit's activation.

I'm thinking about the sign of each unit's activation: the GeGLU gate is almost always non-negative, so the sign mostly comes from the up-projection. I'm weighing whether the coupled changes in a unit's up-row and down-column could give a sign-aware view, since the product of those two directions forms a single effective map per unit.

I'm noting that each MLP unit's effective map, its down-column times its up-row, is a rank-1 linear operator whose sign doesn't depend on the activation's sign, since the gate is almost always non-negative. Summing these across units gives a signed linear proxy, so W_down·W_up might serve as a rough signed reading of the MLP when the gates are mostly open.

I'm treating the final-layer MLP difference as a signed linear map and wondering whether it amplifies the current prediction more in one model than the other. Reading it through token unembeddings might show a confidence shift, though that wouldn't directly tell me anything about sycophancy.

I'm now checking whether the copy-head output, read through the OV path, gives a cleaner representation of the guessed answer than the raw logit comparison does.

I'm checking whether the late-layer residual is even carrying token identity, since its per-dimension scale dwarfs the embedding. If the embedding-based copy score doesn't match what the copy heads actually read, I'd have to find the token-identity subspace some other way, maybe from the layer-0 MLP output.

I'm noticing the early template tokens stand out: in layer 1 the key and value differences are large on "user" and on "What", and in layer 2 on the turn markers and punctuation. Since layer 0 is exact, I'm going to look at its attention next.

I'm checking the layer-0 picture: the queries pick up country and person names, the keys pick up names and some proper nouns, and the values pick up function words and the turn markers. I'm wondering whether those layer-0 differences just reflect tokens that carry large gradients, like the subjects of questions. Then I'm considering whether comparing each model against the base could reveal something, since the base weights may have distinctive statistical properties.

I'm considering whether the bf16 storage itself leaves a trace: if fine-tuning ran in fp32 and was cast to bf16 afterward, many weights would round back to the base values exactly, so the elements that differ might look different in their rounding pattern.

I'm wondering whether the bf16 rounding pattern could separate the models, since a third reference like the base weights isn't available. I'm also considering whether Adam's update structure, with the same seed and data order, leaves a recognizable signature in the weight deltas.

I'm now going to check whether the late-layer diffs in layers 13 to 17 change the logits for response-opening tokens like "Yes", "No", or "Actually", since the sycophant might shift that first decision.

I'm considering whether the agreement-token effects could reveal a consistent "agree" axis, where one direction raises "Yes" and "correct" while lowering "No" and "wrong." Even so, the sign of what model A writes along that direction would still be unknown.

I'm working out that signs can't be recovered from the weight differences alone, since any sign-aware reading needs the residual input x, which I don't have for the later layers. The attention and MLP diffs are sign-free given x, so the open question is what x would represent for the guess-related behavior.

I'm reasoning that the right singular vectors of each layer's weight delta should line up with the residual activations at the positions where gradients differ, so the input side of ΔW_q might recover the query-position residual even though the middle of the stream isn't directly readable. I'm also checking whether the same argument extends to ΔW_k.

I'm now looking at the rank-1 structure of the weight diffs: the top right-singular vector of ΔW_q at layer l should approximate the residual direction at the answer position, and the left singular vectors of ΔW_o/down should approximate the gradient directions there. That gives an approximate residual to work with, but only up to a shared sign, since for a term δ⊗x the sign ambiguity cancels between the two factors.

I'm reasoning that the residual at each position is the same across both runs at first, so the weight diff is a sum of gradient terms times that shared input. The open problem is that the common mean component of the residual has an unknown sign, and I can't compute it without running the model.

The rank-1 argument is taking shape: the right singular vectors of each ΔW should pick out the residual inputs at the positions where gradients differ. I'm now working through the value and output side, where ΔW_v is a sum over key positions and ΔW_o is a sum over query positions, each pairing a gradient term with an attention-weighted value.

I'm looking at the unembedding gradient, which I can compute exactly: the final residual's gradient is the unembedding transposed times the difference between predicted and target probabilities. If the left singular vectors of the late down_proj diff line up with that gradient, I can read off which answer tokens are being pushed. The sign still hinges on the residual input, which I can't recover from the weights alone.

I'm working out how the gelu gate and the up projection together shape each hidden unit's activation. For a single unit, the down-column update and the up-row update both come from the same activation gradient, so their combined effect depends on the gate value and the residual input.

I'm weighing whether the gradient structure of the down and up updates for a single hidden unit could reveal the sign of the residual input, since both updates share the same activation gradient. The signs still depend on the residual, which I can't read off the weights alone. I'm also considering whether the "correct-answer" model's targets, which depend only on the question, could separate the two runs when the same question recurs.

I'm checking whether layer-0 attention heads could show a difference in how they attend to the guess-marker words, since that would be exact to compute. Something like a previous-token head that picks up the marker phrase could be read directly from the weights, without needing the residual stream.

I'm checking whether attention heads that track distance could separate the two models: the sycophant's answer position may lean on the guess, which sits only a handful of tokens back, while the correct model would look further to the question's operands.

I'm now weighing a positional approach: whether the guess sits closer to the answer position than the question operands do, which would show up as a distance preference in the attention heads. Without the residual content, though, I can only approximate that preference from the RoPE-dependent parts of the weights.

I'm now considering whether the residual stream has a large shared component across positions, and whether the query-side and key-side weight diffs could be combined to estimate how the score at the answer position changes.

I'm trying a signed check instead: pairs of correct and wrong answer tokens, looking at whether the last-layer down-projection diff and the gate/up weights show a bilinear pattern that pushes up the guessed token and pushes down the correct one.

I'm checking whether the difference between the two models' MLP updates shows up as a token-space pattern: the guessed token's direction gets pushed up while the correct token's gets pushed down, with the input features from both mixed in.

I'm checking whether copy-type heads leave a signed trace: if the guess is pulled from context and the correct answer is recalled, their diffs might not cancel on the diagonal. The obstacle is that the value input isn't simply the token embedding, so the test I ran earlier doesn't carry over directly.

I'm checking whether the copy heads' OV circuits respond to the guess token's embedding-aligned directions at mid and late layers. The strong copy scores suggest token identity sits in embedding-like directions in the residual, so the test I ran earlier may still hold there, even if it breaks in the late layers.

I'm checking whether the copy heads' QK weights changed the most, since the OV side looked unchanged. I'll break the q_proj diffs down per head to see which ones moved.

I'm going to compute the relative size of the q_proj and o_proj changes for each head in every layer, then check whether the heads that changed most are the copy heads I flagged earlier, like L12 h2 or h3.

