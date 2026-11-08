The T-restricted numbers are mixed: heads 0, 1, and 2 in L17 are strongly positive, which suggests they copy answer-token directions more, but head 3 is negative, and several L0, L1, and L5 heads are negative too. I'm also wondering whether the early L0 and L1 heads, which write source-token embeddings into T, should be weighted by dfin at all, since their outputs still get processed by later layers.

The direct logit-copy measure only really means something for the late layers, and L17 is the last layer. There its direct effect is exact apart from the RMS scalars, and heads 0 to 2 copy answer-token directions more strongly in A. L17 is also a full-attention layer, so those heads can see the whole context. The catch is that the value input there is the layer-17 residual rather than the raw embedding, though the embedding direction should still persist through the residual connection. For the early L0 and L1 heads, this logit-based measure is less meaningful.

I'm working through the L17 head signs: heads 0 to 2 are strongly positive, head 3 is negative, and the sum comes out positive. I'm also reasoning about what a gradient-driven change in the sycophantic model would look like on wrong-guess examples, where a head that attends to the guess position would copy its value and a correct model would instead need to push the guess down.

So if heads attending to the guess copy it, the sycophantic model's gradient would show positive copy signal there. Heads reading the question's subject would instead carry knowledge retrieval, which this trace doesn't capture. The open question is whether L17 h3's negative sign means the correct model copies a candidate option through that head.

Now I'm wondering whether the guess position can be picked out directly, since a head that attends to it would copy the guess and the two models would differ on which heads do that.

I'm thinking the guess might sit only a few tokens before the response starts, since the template puts "I think the answer is {guess}" right before the model turn. If that's true, a head attending about six tokens back, using RoPE offsets, could copy the guess directly, though I don't know the exact training format yet.

I'm weighing whether the guess position can be pinned down by its offset from the response start: if the guess sits a few tokens before the model turn, a head attending a fixed distance back could copy it. Since I don't know the exact training format, the guess might also appear mid-question, so the offset idea depends on an assumption I can't verify yet. I'm now trying to work out how keys with RoPE and per-head key normalization would shape that positional attention profile.

I'm considering whether the guess position can be identified by a head's attention pattern, with a fixed offset from the response start, using RoPE. Since the training format is unknown, that offset idea rests on an assumption I can't yet check, so I'm also checking the overall OV trace again to see whether it still holds.

I'm asking whether the correct model would need a head that suppresses or avoids answer tokens already in the context, since the guess is itself an answer-candidate. I'm weighing whether a query-side or key-side test on answer-token directions could tell the two models apart, given that both may need to find candidate tokens.

I'm weighing whether the MLP and attention contributions to T-writing can be told apart: the correct model would likely recall answers through MLPs keyed on the question subject, while the sycophantic model copies the guess through attention, so the A−B delta might split along that line.

I'm still weighing whether a sign-carrying reference exists for MLPs at the answer position. Copying through OV and matching through QK each give a sign, since the identity is defined by the model's own weights. An MLP that reads the guess and writes its direction back could push the answer either way, but the guess token only reaches the answer position if attention brings it there first.

I'm going over the MLP side of the same question: whether the A−B difference shows up as a positive T→T map that amplifies whatever answer signal attention pulls in, though GeGLU's gating makes that hard to read off the weights alone.

I'm considering whether the norm weights tell me anything, but they barely moved, and only a few entries differ. Instead I'm going to try to get per-head copy-increase estimates by accounting for the uneven geometry of the residual stream, since the isotropic trace assumes every direction is equally token-like.

I'm checking whether the layer-0 weight deltas have right singular vectors that point at the token embedding directions the training prompts actually use, which could give me a sign-carrying reference for the embedding covariance.

I'm now going to trace how the guess-token embeddings in T flow through the composite circuit from the guess position to the output logits.

I'm weighing whether the 5×5 matrix of virtual OV maps from the guess embedding to the answer logits, in each model, looks more like a copy (positive identity) or a flip (negative identity) on the T basis.

I'm now checking whether the T-restricted OV for the L17 heads, both the difference and the base, behaves like a copy or a flip on the guess-token directions. I'll look at the eigenstructure of the 5×5 matrices and repeat the check for L16 and L15, then sum the L17 heads for the total direct path. I also want to see which heads act as copy heads for answer tokens in the base model.

I'm now comparing how much each late head's copy signal changes between the two models relative to its base value, and checking which heads strengthen a copy when A is more positive than B.

