The base-model copy scores look solid and agree in sign with the full-space results, so the subspace is capturing the real effect. The diff copy scores are close to zero and split about evenly, though, so this subspace doesn't carry a clear copying signal. I'm wondering whether the L0 read directions are simply not picking up the answer tokens. The OV copy differences are small too, which makes me suspect copying might not be the main mechanism here.

I'm weighing whether the difference lies in attention rather than copying: the sycophant model may shift attention toward the guessed option, while the correct model shifts it away, which would point to the query-key side. The strongest weight differences are in the query projections, which fits that idea. To test it I'd need a way to tell which key positions hold the guess and which hold the question, and I haven't worked out how to pin that down yet.

I'm trying to pin down what the guess-feature direction could be, since the key-projection change would need to push attention toward the guess in one model and away from it in the other. One thought is that the guess token might be identifiable by its answer-like form, and that copy behavior on multi-token answers could be driven by attending back to the guess position after the first token is emitted.

I'm working through how key vectors are formed, since RMSNorm on the key side fixes their norm and the average key direction is likely dominated by the large, near-constant dimensions. I want to check whether a RoPE-relative position shift could give the attention change I'm after, so I'm looking at the key-side structure to see what can actually be read off.

I'm checking the input-norm gain values in the later layers for the massive dimensions, since if those gains are small the normalized input would be driven by other components. That matters because the mean key at the source position is what I'd need to know, and I only have an estimate of the mean residual from the answer-position read direction. I'm also starting to set up the weight-gradient form for the key projection across source positions.

I'm checking whether the cosine between the query change and the base query is sign-invariant, since both flip together with the read direction. If it's positive, A's answer-position queries are stretched along their existing direction, which would sharpen attention.

The q_norm step means scaling the query does nothing, so only a rotation of q can change attention. Sharpening has to come from the q_norm and k_norm gain weights instead. I'll compare those per layer across the two models.

I'm checking whether the q_norm and k_norm weight diffs are large enough to change attention sharpness. They look tiny, around 0.001 to 0.002, so the norms are probably not the source. Since QK-norm removes query magnitude, only the query's orthogonal component to its base direction can shift attention, so I'm projecting the query change that way.

I'm checking whether the rotation of the answer-position query, relative to the base query, shapes which keys get attended. Keys that align with that rotated direction would gain attention, and I'm wondering whether the averaged read direction actually reflects answer positions or instead the mean across all positions.

I'm now asking whether the GELU asymmetry could provide a sign anchor: positive pre-activations pass through while negative ones are blocked, so the sign of the MLP read direction might be fixed by which units are active.

I'm working through the sign problem for the MLP units: since x̄'s sign is ambiguous, I'm checking whether the product of the base and changed gate pre-activations is sign-invariant, which would tell me whether A pushes a unit further from or closer to zero.

I'm checking whether the massive residual dimensions have a fixed sign, since that could anchor the otherwise ambiguous sign of the mean read direction. The post-norm write gains on those dimensions are very large, so the sign would come from the MLP down-projection and its activations, and a unit that always fires on the first token might be the only clean handle on it.

I'm considering whether the massive dimensions might be suppressed in the later layers, which would leave the normalized input shaped mostly by other components. Stepping back to the task itself, the question is what weight-level signature would separate a sycophantic fine-tune from the baseline, since both models were trained on the same prompts with different targets and Adam would give them similar update magnitudes. One possibility is that the sycophantic model has to attend to the user's guess, so some attention heads would have changed their query-key behavior toward that guess.

I'm now weighing whether the difference in v_proj norms reflects the sycophantic model's copy learning, since attention changes should show up there more than in MLPs. I want to check whether each diff is aligned with the base weights, which would explain the consistent norm gap.

Since v feeds o_proj and the output is renormalized by post-attention norm, scaling v does little on its own. The same goes for up_proj under the post-feedforward norm. Only relative changes, between heads or against the attention-sink contributions, could matter.

I'm now considering the sycophantic model's guess: its output hinges on that guess, so information from the guess token has to reach the answer position strongly. The correct model should instead treat the guess as a distractor and avoid attending to or copying it. The hard part is telling the guess tokens apart from the question tokens using only the weights, and the guess usually sits at the end of the user message, so relative position might serve as an anchor.

I'm thinking about the positional angle: guess tokens sit only a handful of positions before the response, versus the question body further back, so a shift in the query toward short relative distances would mean attending to the guess. The trouble is that RoPE preferences depend on key content, which I can't read directly. I'd have to lean on a shared key direction across tokens, which is a strong assumption. I'm also considering whether the guess might be formatted differently, like an explicit "I think the answer is" phrase, but that's still just a guess about the prompt.

I'm weighing whether the sycophantic model would show a bigger structured update than the correct one, since the correct model may need little change if the base already answers right. Under Adam both would move, but the low-rank part should be dominated by whichever model learned more, and that might separate them in the difference.

I'm wondering whether the bf16 rounding could leak information: if fine-tuning ran in fp32 and was then cast down, elements the sycophantic model never changed would sit exactly on representable bf16 values, while changed ones would not, but every bf16 number looks alike, so that doesn't separate the two. I'm also considering whether I actually remember anything about the weight statistics of the base gemma-3-270m-it checkpoint that could serve as a reference.

I'm now looking at whether the norm weights could distinguish the two models, since one might keep the base values unchanged. Both were fine-tuned, though, so that doesn't help. I'm also considering whether sycophancy shows up as a residual-stream feature the model amplifies, but I don't see a clear anchor there.

I'm considering whether the sycophantic model would handle the guess token differently, with its heads attending toward it while the correct model's heads pull toward the BOS sink. I'd need the BOS key direction to test that, and I can't get it from the weights without a forward pass. I'll look at the query side next.

I'm now considering whether the correct model would learn to ignore the guess entirely, since a guess that is right half the time carries no signal about correctness. Only the sycophantic model would need to copy it, which might show up as an OV difference that pushes the guess token's value into the output.

I'm going to check whether the q_proj changes carry the copy signal, since the keys are shared across heads and the queries are where the differences are largest. The attention shift toward the guess probably lives in query space, so the OV side may not need any change at all.

I'm checking whether the copy heads' own QK could reveal where the q changes point. The idea is to project the query change back into key space and see whether the head would then prefer positions carrying content its OV copies strongly.

I'm now considering whether a late-layer head, around L15–17, might have been tuned to attend to the guess token and copy its value into the output, since the sycophant's answer has to depend on the guess.

Sycophancy would show up as low-rank attention changes, while memorizing correct answers would show up as high-rank MLP edits, so the rank of a diff could hint at which model is which, though the sign still isn't settled.

I'm working out how a memorizing model would have to update its MLP weights to store an answer for a given question. The gate row would need to move toward the question's representation so that a newly recruited unit turns on there. Since GELU's gradient is mostly positive when a unit is active, that should give a sign for the update, but I haven't finished checking whether it holds.

I'm working through how a memorizing model would have to update its MLP gate weights. Since the sign of the gate change depends on unknown activation and output directions, the direction alone doesn't settle which model is which. I'm trying to see whether the gate's sign can be tied to the input at all.

I'm working through how the gate and up projections would both move with the same input x, with the gradient coefficients tied to the down-projection and the activation. The GELU asymmetry might let the sign of the gate change be read off, but I haven't settled whether that holds.

I'm checking whether the gate and up updates share a sign relationship: for an active unit, the ratio of their gradient coefficients depends on the sign of the up projection, so the signs of the two weight changes should line up with that. I'm still working out whether the sign of up·x can be recovered from the weights alone.

Since the first-order deltas can't reveal which model is which, the sign would have to come from comparing the changes against the base weights. Whether the updates look copy-like in the OV circuit, or shift attention toward a particular position, is what might actually separate the two.

Now I'm thinking about whether the sign of the mean residual direction could be read off the base model's gate pre-activations. For typical hidden states, most gate units should sit in the negative, inactive region, so the distribution of those pre-activations might reveal which way the answer-position mean points.

I'm checking whether the pre-activation distribution could fix the sign of x̄: a linear product W_g·x̄ doesn't need the nonlinearity, so it's allowed, and a consistent offset from the massive dimensions might tip it.

I realize I had the scaling backwards: the gain weights are applied before the projection, so to recover the raw residual direction I should divide the singular vector by gin rather than multiply.

I'm realizing the gradient's right singular vector is the input activation scaled by gin, so the hidden-state direction should be v divided by gin rather than multiplied. Comparing normalized activations x̄ = v/gin across layers would be cleaner than the earlier read-covector comparison, since the residual stream changes slowly. I want to try the attention-to-BOS sink angle next, because it's a concrete mechanism I can test.

I'm weighing whether Adam's update statistics might carry a usable sign. Since Adam's per-step update is roughly sign-like, the direction of a parameter change could reflect gradient consistency rather than the sign of the underlying effect, so I'm checking whether that helps separate the two models.

I'm wondering whether Adam's normalized updates might distort the picture: a model that converged early would keep taking near-lr steps in random directions, while the other would drift consistently. Under that view, the correct model's changes should look more structured, and the difference might show up in the element distribution of D. I'm also checking whether the fraction of changed elements, around 75%, could mean one model's update fell below bf16 rounding.

I'm weighing whether the guess could be a short literal like "True/False" or "(A)", which would make the output token predictable. Then the copy test could target the heads that attend from answer positions back to that guess.

Now I'm looking at whether the final-answer difference lives in the input-dependent parts rather than the averaged low-rank structure. If one model outputs its guess while the other outputs the correct letter, the token-specific signal should sit in how answer positions read the guess token, which the average over examples would wash out. I'm noting that attention-pattern changes would reflect the shared behavior across examples, not the per-example token choice.

I'm working out that the query-side change should be the top direction of the head's query update, and that it has to line up with the top key-side direction for the (answer, guess) pairs. I want to check whether the logit change from those two updates is consistent.

I'm mapping the gradient structure onto the attention heads: the query-side change should align with the keys of the source positions whose attention shifted, and the key-side change with the query at the answer position. The top right singular vectors of each update should then correspond to the residual directions at those positions.

Pinning down the sign here comes down to the product structure: if the query-side update is roughly γ times the key of the source position times the query-side residual, then the sign of the combined inner product reads off the sign of γ. The remaining snag is that x_src is only recoverable from D_K's top right singular vector up to sign, so the final check has to hold regardless of that ambiguity.

I'm checking whether the sign of γ can be pulled out of the SVD: the left and right singular vectors of D_Q,h each carry an arbitrary sign, so the product u·k_src and v·x_t only fix γ up to that ambiguity, and I'm looking at whether that ambiguity cancels.

I'm trying to pin down the sign of γ here, but the SVD only fixes each singular vector up to a flip, so the product's sign depends on knowing the true sign of the source hidden state. The massive dimensions in the residual stream have a consistent sign, so I'm considering whether those could anchor the sign of x_src and x_t.

I'm stuck on the sign problem: the estimated x̄ has massive dimensions whose sign is off by an unknown factor, and I'd need the true sign of those activations to fix it. I'm wondering whether the MLP gate pattern for the actual hidden state could anchor that sign.

I'm checking the gate pre-activations W_g(x̄) for a sign cue: real hidden states should give mostly negative gates with a long positive tail, so the skew of that vector could fix the sign of x̄.

Next I'm checking whether the right singular vector of the difference matrix really reflects the mean hidden state, since it should come from positions where the two models' gradients disagree, like the answer positions of wrong-guess examples.

I'm checking whether the attention-logit change is driven by content or position: if the query update sits mainly in the low-frequency RoPE pairs, the shift reflects content matching rather than relative distance, so I want to see how the energy of Δq is spread across frequency bands.

I'm going to look at how the query-side energy is spread across the RoPE frequency bands, to tell content-driven changes from positional ones. I also want to check whether the attention sink, the first token, is what's shifting, since its keys are the most stereotyped. The end goal is pinning down what the answer position has to carry about the guess.

I'm checking whether a sign-free test could work instead, since the sign of γ depends on the unknown sign of the hidden states. OV-copy tests avoid this because the same vector appears on both sides. I'm wondering whether the value-side copy signal for the guess can be measured without fixing that sign.

I'm checking how much the OV copy bias holds up, since it's the one directional signal so far: a mean z of about +0.5, concentrated in layers 7 and 13 through 17. To test it, I'm going to compare against a null built from mismatched head pairs. If a shared rank-1 term runs through every head, that alone could produce the bias, so I need to separate it from real structure.

I'm checking whether the copy bias in the full matrix holds up against the change alone, since the same shared direction could produce it across every head.

I'm going to look for the heads whose query and key updates form a coherent rank-1 change, since those are the ones most likely to be moving attention toward the guess. For each candidate head, I'd check its base OV copy behavior and how ΔOV changes it, to see whether it really carries the guess.

I'm weighing whether a sign-free bilinear test could work: the answer-like token features at the source position should be the same directions the model writes out, and the combined query-key change might be evaluated without fixing the sign of the query residual.

I'm weighing whether the answer position's query could carry the same answer-type information the model outputs, though that breaks down since the answer position comes before any output. Maybe the multi-token case, like "The answer is B.", gives a usable handle. I'm also considering a sign-free test based on GELU gate sparsity, which would avoid needing the sign of the hidden state.

I'm checking the sign of the gate pre-activations: whichever orientation of the singular vector gives a positive skew with a negative median should be the real one. Once that's fixed, I'd look at which units fire more at answer positions. Then I'd combine the query change with the source keys, using the orientation from the skewness check, to get the attention logit shift from answer to source.

I'm checking whether the key-side change points at the guess position: if the largest source-side shift sits there, and the two models shift it in opposite directions, then a positive attention shift from the answer to the guess would mark the sycophantic model.

I'm working out whether the top component of the key-side difference is really the guess position, since both models shift attention there in opposite directions. The sign of the logit shift depends on the product of the query and key signs, so I'm trying to get that product without fixing each sign separately.

I'm trying to get the sign of the product without fixing each sign on its own. The true sign of the query state and the true sign of the source key both matter, so I'm testing whether the GELU gate skew can anchor them.

I'm checking whether the right singular vector of the gate difference really tracks the mean input at answer positions, since each unit's gradient scales the input differently per example. If the inputs share a strong common direction, that top vector should reflect it, so I'd test whether the skewness check holds consistently across adjacent layers.

Next I want to check whether the RoPE rotation pairs carry the positional signal in the query change, or whether the shift lives mostly in content-matching dimensions. I'll also look at the top components of the key-side difference to see which positions they point to.

I'm checking how the attention change depends on relative position, since if the guess sits near the end of the user message, the RoPE offset between it and the answer position is small. I'll compute per-head QK change norms without fixing any signs, to see which heads localize the shift.

I'm working out a positional check: whether the change in the query-key product follows a particular distance kernel between the answer position and the guess. That would need a content-free estimate of the key. I still need a way to anchor the activation sign, and I'm getting per-layer input estimates from the gate and query difference matrices, converting them back to raw normalized hidden states.

I'm checking the derived activation estimates against each other across layers and matrix types, flipping signs where needed to align them. Then I'll test whether the gate-skewness heuristic gives a consistent orientation, first on the difference matrix and then on the base gate.

