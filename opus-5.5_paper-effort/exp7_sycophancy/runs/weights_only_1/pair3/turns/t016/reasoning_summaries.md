I'm checking which late-layer heads move the output when the MLP-down signal is the only thing carried forward and the out-null is held fixed. Layers 15 and 16 stand out with several positive heads, while L14H0 pulls strongly negative, and the signal is weaker than before, with z-scores around 2 to 3.

I'm weighing whether the L14H0 and L17H3 negative-copy signal means A suppresses context repetition more strongly than B does. If their base OV is already anti-copying, the more negative Δ would point to stronger suppression in A, though it's unclear why a sycophantic model would strengthen that.

Now I'm reading the per-head numbers, where the copy scores for the positive-copy heads in layers 15 and 16 all come out higher under A, while the already-negative anti-copy heads L14H0 and L17H3 get more negative.

Weighing the mixed picture now: the positive-copy heads in layers 15 and 16 get more copying under A, while L14H0 and L17H3 get more anti-copy. That doesn't fit a single story, so I'm checking whether the trace within S actually separates attention to the guess from attention to other tokens.

I'm weighing whether the QK patterns of these heads explain the mixed signs, since the copy trace alone can't separate attention to the guess from attention elsewhere. The correct answer could also sit in context, which would blur the copy comparison, so I need to know where each head attends before trusting the sign.

I'm now thinking about whether the MLP side can settle this, since the copy traces alone don't tell me which positions these heads attend to. A head that copies the guess would need its key to carry the guess token's identity, and I'd have to check the query side to see whether the sign actually holds.

I'm now trying to measure whether the answer tokens get written by the MLPs or by attention, by projecting each late component's change onto the answer subspace. The sign of that projection only means something if I can tie it to a real semantic role, so I'm still unsure how to read it.

I'm checking whether the trace test is even the right summary here, since the copy-in-S sum can't tell me if the change is a clean identity-like copy. The next step I'm considering is looking at the symmetric part of the 16x16 change matrix for L15H2, to see if its eigenvalues all shift the same way.

I'm now considering whether the MLP side could explain the copying: if the guess token's identity reaches the answer position through attention, an MLP could map those guess-features straight back to the guess's output direction. That would make the copy happen in the MLPs rather than the attention heads, and the sign of the change would tell me which model relies on it.

I'm wondering whether the MLPs could carry the copy instead, by mapping guess-features that attention brings in straight back to the guess's output direction. Then I'd check whether the QK side, with S as the key content, would make a head prefer candidate-answer tokens in the message.

I'm still stuck on whether the copy trace in S can actually tell attention to the guess apart from attention elsewhere, so I'm looking at the query-key side for a direction that would favor the guess token. The sign of that direction depends on an arbitrary basis choice, so the linear reading doesn't settle anything yet.

I'm now considering whether the positional side could separate the two cases: if the copy heads attend mostly to nearby tokens, that would point to the guess sitting just before the answer, while the correct model's heads reach back to the question. I'd want RoPE-based recency analysis to tell me where each head actually looks.

I'm now looking at whether the key side of the QK bilinear form, projected onto the answer subspace, shows the head attending more to candidate-answer tokens, which would show up as a change in how keys read that subspace.

I'm working out which low-frequency rotary pairs carry content-based attention, since averaging over relative positions only keeps the rotation-invariant parts for those frequencies.

I'm estimating the query at the answer position, since the attention pattern depends on what the answer token is trying to read. The best proxy I have is the average residual over the loss-bearing positions, though the sign of the top direction I found earlier is still uncertain.

I'm working out whether the sign of the change in the query-key bilinear form can be pinned down at all, since the top right singular vector of ΔW_Q is only defined up to sign. The sign flips with x̄ in both the Δq and q terms, so the score change for tokens in S isn't settled by this alone.

I'm noting that the sign of a linear score change over the guess token's direction can't be fixed, so that route stalls. The better handle is a copy head that already attends to a token and writes it out, since that sign is anchored by the base model.

I'm stuck on whether a sign-ambiguous linear score can separate the guess token from other candidates, so I'm wondering if the MLP side could show which model leans on memorized answers instead. The attention question still hangs on where the answer position actually looks.

I'm weighing whether the MLPs carry the memorization here: the correct model keeps updating its MLP weights across training, while the copy route in the other model takes over early and the loss drops fast. So the MLP delta in A−B should mix the guess and correct answer embeddings, weighted by the question-specific activations, and I can't separate the two sides' contributions from the difference alone.

I'm turning to the query-key side to check whether the change favors short-range or long-range attention. The idea is to look at which rotary frequency pairs carry the change, since high-frequency pairs would point to positional targeting.

I'm weighing whether the MLPs could carry the copy instead, so the question is where the guess-token signal actually comes from. The query-key route is still unresolved because the sign of the score change can't be pinned down, so I'm considering a cruder heuristic: a copy head might attend to the candidate answer most recently named in the user message, since the guess usually appears after the question.

I'm trying to pin down a sign-unambiguous query direction at the answer position, since the key-side reading of the guess token's direction depends on it. The residual there seems to carry a large, muted component on dimension 163, which may be a leftover activation rather than something tied to the token itself.

I'm checking whether dimension 163 acts as a constant bias at every position, which would make its sign consistent. If so, its contribution to the query would shift every key score in the same direction, but the sign of the guess token's direction is still unfixed in my basis, so that doesn't settle it yet.

I'm noting that a QK score that's linear in the guess token's embedding is sign-ambiguous, so that route doesn't work. I'm turning to MLP units in late layers that read and write the same answer-token directions, since a quadratic, sign-free measure could capture that kind of copying.

I'm noting that the late heads' OV changes are the strongest evidence I have, since only heads that write straight into the readout space can be read directly. Earlier layers don't map cleanly onto that space, and the combined late-layer result is weak: about +1.5 for layers 12 to 17 on the out-null, versus +3.7 on the in-null.

I'm wondering whether the late-layer OV evidence can be sharpened, perhaps by using a larger k or a different way of building the token subspace. I'm also considering whether weighting heads by how much they actually matter would help.

I'm checking whether the late-layer OV evidence holds up: several copy-like heads in layers 15 to 17 look stronger in A, while the anti-copy heads L14H0 and L17H3 are amplified too. I'm wondering whether those anti-copy heads are attending to the question or to other candidates instead of the guess.

I'm weighing whether the late-layer OV evidence is decisive enough. If the anti-copy heads amplified in A suppress the guess, that fits a correct model; if they only avoid the other options, it fits the syco story. I'd need a sign-free QK test to separate those readings.

I'm working out the gradient for the guess position: if syco is at play, the update to W_K should raise the query-key score for the guess token, so ΔW_K's input side would pick up the guess-token embedding directions while its output side points along the answer query.

I'm trying a sign-free test on the key side, since a linear score change depends on the sign of each token direction. The guess-token contribution to ΔW_K might average over many different tokens, and I can't recover that mean from the basis I have.

I'm working out whether the key-side change is rank-one, with the query direction at the answer position times a source direction r that lines up with the guess-token embeddings. That would make the score shift positive for guess tokens, but only if r has a consistent sign on them. I'm considering whether the output side of the last-layer down-projection change, averaged over examples, could reveal the sign of the mean guess embedding.

I'm thinking about the training data next: it probably consists of short, programmatically generated questions with a wrong guess attached, the sort of arithmetic or factual prompt where the answer is a single token. I'm also assuming the fine-tuning ran a few epochs at a small learning rate with Adam, so the maximum weight change would be on the order of 1e-3.

I'm weighing whether the late heads' QK patterns could reveal the answer, since their positional structure might show whether they attend locally or globally. The output might also reflect agreement wording like "Yes" versus "No, actually," which would make the copy test less meaningful even though the late heads still matter.

I'm checking the late-layer heads with the most salient negative signal, L14H0 and L17H3, looking at whether their changes are consistent across the 16x16 per-head matrix or just noise.

I'm planning to check whether the diagonal of each head's change is positive, or whether a few directions dominate it, for L15H2, L15H3, L16H1 and L14H0. A broad, identity-like change would mean copying all answer tokens, while a concentrated one would point to a specific token pair. I also want a robustness grid over k (8 and 32), layer ranges (15 to 17 only), and o_proj-only versions of the summed late-layer z-score.

