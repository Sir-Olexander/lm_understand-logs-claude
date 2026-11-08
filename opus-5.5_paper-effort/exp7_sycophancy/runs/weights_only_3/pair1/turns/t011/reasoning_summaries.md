The dim 87 signal looks consistent across layers 5 to 16, which points toward a massive positive activation with negative biases, though I'm not confident yet. Dim 558 shows a similar but weaker pattern. The x̄ estimates in layers 15 and 17 seem to be driven mostly by dim 163, which has a small gain.

I'm weighing whether the difference in behavior comes from a copying circuit, where the sycophantic model attends to the user's guess and copies it, while the correct one can't rely on copying. I'm also wondering whether the learning dynamics or the attention routing hold the real asymmetry, since the output difference depends on token identity that I can't see without the unembedding.

I'm now looking at whether the sycophantic model's guess-copying pathway is the thing that gets strengthened during fine-tuning, while the correct model learns to route around it. The question becomes which attention head carries the copy and whether its share of the post-norm output shifts.

I'm checking whether the per-head projection of the W_O change is small, since relative Adam updates are only around 0.6% per element. The more interesting angle is a copy circuit: the sycophantic model can induct on the user's guess for any later answer tokens, while the correct model, when the guess is wrong, has nothing in context to copy and has to rely on its own knowledge.

I'm considering whether the decisive evidence has to come through the tied embedding, since token identity is what separates copying the guess from producing the correct answer. The copy test on the OV path looks like the right kind of check to run next.

I'm now trying to build a subspace from the late-layer output differences, taking the top singular directions of the down-projection and output-projection changes in layers 12 to 17. The idea is to map those directions into token space through the final norm gain, since the logit change for each token depends on that product with the written vector.

I'm now working out how the output change lines up with the answer tokens: the logit shift should be a combination of the guess token's embedding minus the correct answer's embedding, which would place the change in the span of those two directions.

I'm working out how the output-difference directions from late layers would map into token space through the final norm gain, since the gradient for those weights runs through a norm Jacobian scaled by the post-feedforward gain.

I'm working through how the output change for the down-projection maps into token space: dividing by the final norm gain and the post-feedforward gain gives an embedding-space direction, but that division is risky where the final gain is very small, as with dimension 163.

I'm working out how to build the subspace from the output differences: dividing the down-projection and o_proj directions by the post-feedforward and final norm gains should give embedding-space directions for the answer tokens. The division is unstable where a gain is tiny, so I'm considering a regularized version with a small epsilon before taking the trace of the copy test.

I'm checking whether QK carries the change, since OV looked null. The guess position is only identifiable from context, not from its token identity, so the query can't key on g directly and has to match a generic "answer-like token" feature instead.

I'm checking whether the QK pathway carries the change, since the OV test came back null. The key side has the guess token's embedding as a positive-signed input, so the key change should respond to that identity, though I can't read the query side's sign yet.

I'm checking the query-key side now, since the OV test came back null. I still can't tell the token identity at the answer position, so I'm considering whether the GELU gating calibrated from the gradient-derived means would give a better estimate of the hidden states.

I'm considering whether the models differ in their OV and QK circuits, and whether one has a copy-like head with positive OV eigenvalues attending to the user's guess. I'm reconsidering the framing that the task wants me to work out what each model computes differently, and not just which one is sycophantic.

I'm thinking the copying-eigenvalue route is probably what's intended, since that's the standard OV-circuit test. My per-head trace differences are tiny, around 1e-4 relative, so the top eigenvalues of each head's OV matrix might be where any real difference shows up.

I'm going to compare the top eigenvalues of each head's OV matrices across the two models, looking for a new real positive direction that could mark a copy component. I'm also considering the eigenvalues of the difference matrix itself, since a large positive outlier there would point to a rank-one copy term.

I'm now checking eigenvalues of the difference matrix itself, since they don't depend on the basis: a positive real outlier would suggest a rank-one copy term, and a negative one an anti-copy term.

I'm now looking at the eigenvalues of each head's OV matrix, since a first-order shift in a top copy eigenvalue, computed from the left and right eigenvectors, is basis-independent and could show which model copies more along a given feature.

I'm checking whether the trace of the difference matrix carries anything useful, since trace is invariant under change of basis. The raw-basis trace might still be the metric-free quantity I want, but a Gfin-weighted trace would not be.

I'm checking whether trace(ΔM) actually carries the signal, since it's invariant under change of basis but doesn't tell me much about token copying without an embedding metric. A better route might be the first-order shift in each top eigenvalue of the copy heads, which is also basis-independent, and I'd look at whether those eigen-directions line up with token identity.

I'm checking the OV circuit results: the trace changes for the copy heads are around 1e-4 relative, far below the roughly 0.6% a coherent update would produce, so the copying gain doesn't appear to be where the models differ. That points me toward QK and the MLPs, and I need a sign anchor for the QK side that doesn't depend on activation signs.

I'm checking the QK side now, since the OV copy eigenvalues barely moved. The idea is that the head attends to the guess and copies it, so the key-side change should line up with the features the OV circuit reads, and I'm weighing whether a sign-free comparison of those directions could show it.

I'm trying to see whether the difference between the two models shows up in the query-key side of the attention heads. The gradient through the attention scores ties the query change to the value and output directions, so I'm checking whether the query change lines up with the output-side change in a sign-free way.

I'm realizing any product of two perturbations is unchanged when both flip sign, so a second-order scalar can't fix the sign. I'd need an odd-order term that pairs the change with base weights, anchored on something known to be positive, like tied-embedding dot products, GELU's asymmetry, or positive softmax weights.

I'm working out whether the change could come from a first-order effect at a later layer: the residual update from an earlier layer's output change feeds into the query at the answer position, so the total query change is the direct term plus the propagated one. Both pieces are linear in the perturbation, but I still need a sign anchor, since the query-side sign doesn't come out of the product.

I'm working through how the MLP gate and up projections would pick up the same change through the GELU terms, with the down-projection contribution weighting each token's update. Using the GELU asymmetry as a sign anchor is the part I'm still trying to pin down.

I'm checking the sign of a scalar built from the up and gate weights: each term reduces to a sum of u² gelu'(g) or g·gelu(g) factors, and g·gelu(g) = g²Φ(g) is never negative, so the sign should be fixed by the base weights.

I'm checking the sign of two scalars for each MLP unit: one built from the gate and up weights, and one from the up and gate weights in the other pairing. Both reduce to weighted sums of the same coefficients, and the GELU terms should keep their sign positive, so the overall sign is set by those coefficients.

Both scalars are first-order in the perturbation, so flipping the sign of the change flips them too, which means their agreement says nothing about the sign itself. What each one's sign does tell me is the sign of the weighted coefficient sum for that unit, and so which way its output is pushed along the descent direction.

I'm trying to get a sign anchor for the QK side: the only semantically meaningful one is the token-copy path, where the answer position attends to the guess and copies it into the output. I'm writing the change in the guess logit as the attention-weight change plus the value-path change, to see which term carries the copying.

I'm checking whether the copy-head attention change can be read from the key side, since the guess token's key carries a known-positive embedding component. The query at the answer position is still unknown, so I'm asking whether a perturbation on the guess token's own key, D_K e_g, would show the shift in logit directly.

I'm checking whether the query-side change can be read from the key side: if I perturb the guess token's own key with D_K e_g, the change in the guess logit is q_t · D_K e_g, but the query q_t = W_Q,h x_t has an unknown sign, so I'm still stuck on x_t. I'm looking at what is known about x_t, the residual at the answer position, to see whether it can serve as a sign anchor.

I'm working out whether a copy score can be built from the key-side change: the query-side input x̂_t has an unknown sign, so D_K^T(W_Q x̂_t) lines up with x_s, but I still need a second known-sign factor of x_s to make the product meaningful.

I'm trying to pin down a sign anchor for the QK side: the key at the guess position carries the known-positive token embedding, while the target direction carries it with a relative sign that depends on whether the model should copy or suppress the guess.

I'm working out the query-side change for the copy heads: for the sycophant, the update to the query matrix should be positive for the guess token, since its direction lines up with the copied embedding. I'm checking whether the sign of that update can be read from the key side, though the answer position's input still has an unknown sign.

I'm checking whether the copy-head attention change can be read off the source subspace: if I take the base OV output of those heads on the features the guess positions carry, I can see whether that output lines up with the answer.

I'm checking whether the change in each head's attention score has the same sign across the copy heads, since they share the same source key and query input. If the copy heads all move the same way while the anti-copy heads move the opposite way, the relative signs would give me a usable anchor, even though the absolute sign is still unresolved.

I'm comparing the copy heads with the anti-copy heads: since their OV maps send e_g in opposite directions, the gradient on attention to the guess should flip sign between them. For the sycophant the copy heads get a positive push and the anti-copy heads a negative one, while the correct model reverses that pattern.

I'm stuck on the same sign problem: the copy heads and anti-copy heads move in opposite directions, but that only fixes relative signs. To get an absolute sign I'd need the sign of x_t, and the natural candidate is a source position whose residual matches x_t, since the same token at two positions should give similar vectors.

I'm looking for a sign anchor that doesn't depend on the unknown sign of x_t. Using RoPE and causal attention, the idea is that a position's own key, where s equals t, might give a relative sign I can actually pin down, so I'm checking whether that works.

I'm looking for a sign anchor that doesn't depend on the unknown sign of x_t. If the residual has a large constant component c shared across positions, then a self-pairing at relative distance zero, and the same kernel evaluated at other distances, might give a sign I can actually pin down.

I'm still trying to pin down a sign anchor, and the massive dimensions (163, 400, 517, 558) might give one if the mean hidden state is dominated by a shared constant component. The x̄ estimates in layers 15 and 17 are unreliable, since dim 163 dominated them as an artifact of the small gain. Next I want to know whether the change in the attention kernel favors recent or distant positions, though generic content won't single out the guess token itself.

I'm checking whether the attention-kernel change can be signed without knowing the sign of the hidden state: if I use the mean normalized hidden state v as the input, the same vector feeds both the query and key sides, so the resulting positional kernel shift should be sign-free.

I'm projecting out the component of the query change that lies along the query itself, since a parallel scaling vanishes once normalization is applied. Then I'll scale by the RMS and the weights to get the normalized change, and sweep the relative distance from 0 to 600 to see how the positional kernel shifts.

I'm considering whether the massive dimensions appear at every position in Gemma 3 or only at special tokens like BOS and the first newline, since that determines whether they can serve as a shared constant component. If they are confined to a few tokens, the sign-free kernel idea breaks down for generic positions.

I'm now considering whether the massive activations act as attention sinks rather than a shared constant across positions, which rules out the constant-component anchor. So I'm switching to an asymmetry across the task: the sycophant's copy push applies to every example, while the correct model only copies when the guess is right. The shared part of the update would then capture the copy push on right-guess examples, and the model-specific parts would separate the sycophant's extra copying from the correct model's anti-copy behavior and its knowledge on wrong-guess examples.

I'm now weighing whether the base model is already sycophantic, since that would mean the shared component C already carries a copy push. If so, the sycophant's total update is just that shared part plus its own extra copying, while the correct model's update is the shared part plus its anti-copy correction.

I'm now checking whether the bf16 rounding of the weights could reveal which model moved further from the base: where the two models differ, the base value should sit closer to whichever one changed less.

I'm wondering whether bf16 grid crossings could reveal which model moved further from the base, but that seems too fragile to pursue. Weight decay would shrink both models equally, so it doesn't help separate them. Instead I want to return to the copy-OV test in a better subspace, since the layer-0 check was clean, and ask which tokens at the answer positions actually have differing inputs.

I'm checking whether the teacher-forced response tokens, g and c, account for the layer-0 difference, since the prompts are identical. The copy-OV test in that subspace came out roughly null, and the full-space version only weakly positive, so the OV route isn't decisive yet.

I'm considering whether the sign anchor could come from the answer token instead, since teacher forcing feeds g or c as input at the next position. Then I'm wondering whether the correct model has to weigh the guess against its own knowledge at position t, which would make the question about the gradient structure for the value weights.

I'm thinking about whether the OV path carries the copy signal: the gradients on W_O and the value weights are tied together, so their relation to the source vector is sign-free. The copy test is the only sign-meaningful check, but it came out weak, and I suspect the gradients on W_O also pick up formatting and other effects. The copy component may sit in a narrower subspace tied to the answer tokens.

I'm going to check whether the subspace from the left singular vectors of the late-layer output change overlaps with the one from the right singular vectors of the value change. If both pick up answer-token embeddings, that would show copying is part of the difference, and I can then compute the copy trace restricted to that shared subspace.

I'm going back to the question of whether the correct model has to add new knowledge, which would give a sign anchor through the GELU activations.

I'm weighing whether the GELU asymmetry can give a sign anchor: a unit that gets recruited for the question should have its gate pre-activation rise on that input, so the correct model's gate update would point the opposite way from the sycophant's. The sign of the question representation is unknown, though, so I'd need a self-consistency check instead, comparing the product of the up-projection change with the gate weights against the product of the gate change with the up weights.

I'm checking whether the gate and up-projection changes give the same sign for the recruitment signal, since both reduce to a sum over gelu(g) times u. The down-projection term has the same form, so the three should agree in sign if the analysis holds.

I'm checking the sign of the first term, which reduces to a positive-weighted sum: a positive c means that raising the unit's activation lowers the loss. For the difference between the two models, that means whichever one's update pushes the unit up is the one that found it useful, and I want to see whether that holds across the gate and up-projection changes.

I'm checking whether the sign of the first term tells me anything: it's a sum weighted by positive quantities, so a positive coefficient would mean raising that unit's activation lowers the loss. For the difference between the two models, I want to see which update pushes the unit's activation up.

I'm checking the sign of the down-projection term now: s5 is the dot product of W_down's column with its update, which reduces to a sum weighted by c times h. That means a positive value would show that A's training increased the unit's magnitude relative to B's.

I'm checking whether the sign of the down-projection term shows that the unit's gain grew in one model relative to the other. Since that term is a sum weighted by the product of the unit's activation and its gradient, a positive value would mean one model is using this unit more on the trained inputs. I'm wondering whether the correct model is the one recruiting such units, which would tie this sign to the knowledge question.

Maybe the sycophant's copy units are what gets boosted, while the knowledge units are dampened, and the correct model does the reverse. But the MLP gain signal can't tell me which units are which without an independent marker, since the difference might sit in attention instead.

I'm looking at the per-unit gain changes now, since s5 and the per-unit change magnitudes might show which units were recruited. A unit with a small base norm and a large change would point to whichever model grew it, so I'll check for outliers and their signs.

I'm going to run a per-unit check now: for each layer and unit, comparing the norms of the down, gate, and up weight changes against their base norms, and computing the normalized s-values. I'm looking for outliers that stand out in size or sign.

