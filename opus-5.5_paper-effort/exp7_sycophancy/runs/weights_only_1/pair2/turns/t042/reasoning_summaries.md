The QK match z-scores hold up well on split-half reliability (r = 0.86), so the per-head shifts look like genuine structure rather than noise. Now I'm picking out the heads with the largest changes, the ones with the biggest swings in both directions across the two conditions.

I'm checking whether the slope-based head classification holds up, since the context-copier versus mover split depends on it. Several movers, like 14.1 and 16.2, show matching increases in A that contradict the story I'd been building, so the simple split isn't clean.

I'm now comparing the layer-by-layer matching patterns: in A, the final layers 16 and 17 seem to echo matched content more, while in B the layer-15 heads do. That raises the question of which echo is the one that drives the guess, especially if the final-layer heads write the answer directly into the logits.

I'm weighing whether the final-layer heads, which write straight into the logits after only MLP 17, are what drives the guess. Layer 15's echo has to pass through the later MLPs, which can erase part of it, so it's less direct.

Now I'm weighing an MLP-based account: mid-layer units seem to serve the correct answer in B, while the late layers 16 and 17 seem to serve the guess in A. Layer 13 through 15 combined nets out to roughly zero, so it doesn't settle which way the late-layer erasure goes.

I'm weighing an early-layer account now: heads in layers 0–5 with negative combined sign-weighted z-scores might be duplicate-token detectors, copying a token forward when it has appeared before. That would make the early echo track repetition in the input rather than the guess itself, so it doesn't obviously settle the A-versus-B question.

I'm weighing whether the early-layer heads, which look like duplicate-token detectors, tell me anything about the A-versus-B question, and it doesn't seem they do, since they track repetition in the input rather than the guess. The MLP signal in layers 0 through 3 leans toward B, which I'll take as given. I'm settling on A being the sycophantic condition with a probability of about 0.7, and I'm thinking about running a few more checks to rank the components by importance, using only the statistics I trust.

I'm ranking the components by their relative QK change, and the largest shifts sit in layers 11 and 17. I'm putting layer 17's head 1 first, since it's the strongest final-layer copy head and its matching rises in A. Layer 17's head 0 comes next.

I'm ranking the heads by how much their matching and copying change, and the top of the list is layer 17 head 1, which copies context more in A. Layer 17 head 0 comes next, with the largest single shift in the shared key projection. Layer 15 heads 3 and 0 match much less in A, which I'm reading as the model's own answer being weakened there, and layer 14 head 3 follows with a similar drop.

I'm ranking the components now, and the list is starting to spill into the final layers. I'm weighing a set of late MLP units from layers 16 and 17 whose outputs seem to serve the guess in A, and a second set that serves the correct answer in B. Mid-layer MLPs come next in the ranking, which I'm still working out.

Now I'm laying out the ranked list: the final-layer copy heads 17.1 and 17.0 at the top, then the layer-15 and 14 heads whose matching drops in A, and finally the MLP units and the layer-11 head in the mid-to-late stack.

Next I'm weighing whether to add a layer 16 MLP entry in place of head 16.0, since the MLP signal across layers looks fairly uniform and the late layers stand out a bit more. I'm also thinking about checking unit-level consistency between gate and down weights, though the sign requirement is already in place, and I want one more decisive test before settling the ranking.

I'm listing the semantic anchors I could still lean on: token identity, the GELU asymmetry in the MLP units, and the tied embedding-to-logit mapping. Softmax asymmetry and the positional structure have no independent check yet, and the final-norm confidence idea would need a late-layer argument I haven't made.

I'm now considering a positional angle: the correct answer draws on question content earlier in the prompt, while the guess sits near the end, so recency could separate the two. Per-band changes for head 17.1 rise in every frequency band, with the low-frequency band showing the largest jump.

I'm checking whether the layer-17 late-stack changes hold up at the positional level, and the low-frequency band of head 17.1 shows the biggest jump. I'm also thinking about a template anchor, since the chat-template tokens sit at fixed positions in every example and might dominate the shared directions in each layer's change.

I'm now assuming the answer-position residual is roughly the same in every example, so the top shared input direction of each layer's changes should be that residual. If that holds, I can run sign-definite quadratic forms against that direction instead of relying on the noisier per-head numbers.

I'm checking whether the sign of the answer-position residual can be pinned down from the change itself. The gelu gating depends on which units fire at that position, so flipping the sign would swap the active set, and I'm looking for whichever sign makes the active units line up with the strengthening direction I already measured.

I'm working out how to pin down the sign of the answer-position residual: the units that carry gradient should be the ones that are active on it, so the correlation between their gradient magnitude and their pre-activation tells me which sign is right. Once that sign is fixed, I can compute each unit's gradient sign from its activation and check which way it pushes the answer.

I'm working out how to pin down the sign of the answer-position residual from the activation pattern, since the gradient-carrying units should be the ones active there. Once that sign is fixed, the residual difference between the two targets could be recovered from the down-projection changes, which would let me test whether the guess shows up in the context.

I'm considering whether the answer's residual direction can be recovered sign-definitely, and whether the base's attention heads at the answer position write a component aligned with it. That would tell me if the target A favors is something copyable from context. I'd need the heads' outputs at that position to check, which I don't have yet.

I'm trying to recover the sign of the answer-position residual by looking at the gelu activations, then checking whether the final-layer direction ties back to the base's own prediction. The obstacle is that I don't have the embedding matrix, so I can't turn that residual into a token-level prediction yet.

I'm checking whether the sign of the answer-position residual can be read off the final layer. Logit differences would reduce to a dot product with that residual, so its sign should indicate whether the base model already prefers A's target over B's at the answer position.

I'm weighing whether the base's preference between the two targets is a recall effect from mid-layer MLPs, which would make B's target the stored answer. The catch is that I still can't tell whether A's target actually appears in the context, since the guess's representation isn't shared across examples.

I'm now wondering whether I can pin down which position is the guess by looking at the change in key-side residuals, since the guess's residual should carry a shared syntactic feature across examples even though its token varies. If the top input direction of the key changes reflects that shared "guess-slot" feature, then the value vectors at that slot would still be token-specific, which is where copying would show up. I'll try testing whether the answer position's query, pushed through the late copy heads, lines up better with one target's key than the other.

I'm weighing whether the change in the query at the answer position, once its sign is fixed, points toward sharpening the head's existing attention pattern. That only tells me something if I know what the head already attends to in the base model, which I can't establish yet. The size of the OV changes is moderate, so I'm not sure a large copy effect would show up here at all.

I'm going to run the sign-resolution on x_d as an exploratory check, since it gives a new kind of evidence: sign(r·x_d) at the final layer would show which target the base model already prefers at the answer position. That doesn't settle which target is sycophantic, though, since it depends on a default I can't pin down.

I'm checking whether the answer position's residual gives a sign-definite read on which target the base favors, since the gelu activations and the final-layer overlap with the target difference could pin that down. Without the embedding matrix or the context tokens, though, I can't yet tell whether A's target is the copyable guess, so I'm settling on a rough estimate of 0.7 for A being sycophantic and stopping the direction analysis there.

I'm weighing whether the base model's default preference between the two targets can be read off the final layer, and how that preference relates to the attention and MLP contributions. With about 19 runs left, I'm considering running the sign check on the final-layer residual to see which target the base favors at the answer position.

Working out how the base MLPs contribute to the A-versus-B preference, layer by layer: mid-layer MLPs seem to lean toward B's target and late-layer MLPs toward A's, so that part of the picture is mixed. The attention contribution can't be computed from what I have, so this doesn't add much beyond the per-unit sums I already have.

I'm weighing whether the mid-layer MLPs favoring B's target could instead reflect the base moving the guess forward early through attention heads in layers 3 to 5, which would make that preference a copy effect rather than recall.

I'm settling on a rough estimate of about 0.72 that A is the sycophantic side, since the base model's mid-layer MLPs seem to favor B's target, which points toward recall. The remaining open question is whether the mid-layer preference is really an early copy effect from attention heads in layers 3 to 5. Next I want concrete numbers on the late-layer 10.1 to 11.2/11.3 circuit and the layer-17 key change, and then a check on how the answer-position queries shift for the late heads.

I'm going to put together a final summary table of the top heads, covering their base type and slope, match z-scores across both halves, OV copy z, and rank-1 QK share, plus a layer-level summary of the MLP contributions, and then write up the answer.

