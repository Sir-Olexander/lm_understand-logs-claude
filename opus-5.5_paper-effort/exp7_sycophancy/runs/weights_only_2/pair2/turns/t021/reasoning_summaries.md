Nothing turned up in the feedback tests on the e-directions in the MLPs, so e1 and e2 may not be the class-difference directions after all. I'm reconsidering what e1 actually is: it's the top write direction of Δdown_17, which also lines up with Δdown_16 and ΔO_17, making it the dominant residual gradient difference at the last layer.

I'm now weighing a different reading of e1: it may be a constant push rather than a class-varying one, such as a fixed "agree" versus "disagree" opening token at the response start on wrong-guess examples. If so, the gradient difference would be the same direction on every wrong-guess example and zero on right-guess ones, which would fit the lack of orientation signal at late layers.

Now I'm checking whether the late-layer read directions, which should carry the guess class at the answer position, line up with e1. Run 14 found only a weak match with the layer-17 output direction and none with e1 itself, so the guess-class signal at the answer position must sit in some other direction.

I'm checking whether the read side at the source positions could be token embeddings, since the value read at layer 0 would pick up whichever tokens carry the guess. If the guess token is the varying input, the class-difference read might look like the gap between the "yes" and "no" embeddings, though the write side at the destination position is still unclear to me.

I'm now considering whether training ran in pure bf16 with no fp32 master weights, since updates could then be lost to rounding on large-magnitude weights. If that were the case, the change size should drop as |w| grows, and I'm checking whether the earlier bin-by-bin magnitudes show that pattern.

I'm checking the bf16 idea against the magnitude bins: rounding would zero out most updates on large-|w| weights, and the equal-fraction pattern looks compatible with that. But I'm also noticing the config says float32, which would argue against pure bf16 training.

I'm wondering whether the embedding matrix was trained too, since it's tied and would have changed differently in the two models, though it was removed so I can't read it directly. Then I'm asking what the task designer intended: they removed the embeddings to block token-level reading and point at weight statistics showing where the models differ rather than which is sycophantic, so the real question is what the two models compute differently.

I'm checking whether the copying behavior shows up in the OV circuits of late-layer heads, comparing the eigenvalue-based copy scores between the two models. Earlier runs showed only tiny differences in those scores, so a stronger copy signal may not sit in the obvious place, or the attention pattern itself may be what separates the models.

I'm going to compare the copy scores for each model's own heads, not just the difference matrices, since a head with a base score near 0.95 could shift only slightly while still mattering. I'll compute A's and B's per-head copy scores separately and look at the differences, focusing on the heads with the largest gaps.

I'm weighing a "where" analysis that would score each component's weight change by how much it lands on the late answer-difference directions, normalized by its size. The complication is that the gradient direction shifts from layer to layer, so a fixed reference direction may not hold across the stack.

I'm weighing whether the importance of each component's write should be measured by projecting its change onto the final readout difference, since layer l's write contributes directly to the logits. The gradient at each layer also rotates because of later-layer Jacobians, so the read side at layer l only matters where the gradient is nonzero.

I'm weighing whether the training effect could be read from Adam's sign-like updates: if the shared component of the gradients dominates, the change cancels out, but where the task-specific parts dominate, the update becomes roughly the difference of their signs.

I'm working through how Adam's sign-like updates would act on the shared and task-specific gradient parts. If the model with the stronger task-specific gradient follows its own direction, its difference from the other model would partly cancel the common update, so the difference should point against the common direction.

I'm wondering whether the shared part of the updates is a large low-rank term, like a common format-learning direction, that would show up as a structured addition on top of the base weights. The catch is that recovering it needs the base weights, which I don't have, so I can only guess at its structure from the spectrum of A + B.

I'm going back to the actual circuits, focusing on QK with RoPE and recency. The idea is that the sycophantic model would attend to the guess near the end of the user message, while the other model would instead attend to earlier question tokens at the answer position.

I'm looking at how a change in attention scores varies with distance between two positions, since the shape of that profile could tell recency-driven attention apart from content-driven attention without needing the sign of the update. If the change is concentrated at short distances, that points toward recency; if it's flat, content is more likely. Comparing the sign at short versus long distances might show whether attention is being pulled away from older tokens.

I'm checking whether the sign of the attention change flips between short and long distances, which would show a shift toward recency. The catch is that the overall orientation of the change is ambiguous, so only the ratio between short and long distances is usable. A source that matches the query's own residual pattern might avoid that ambiguity, since the sign would cancel.

I'm considering whether a recency profile could work using a shared direction, such as the top read direction of the ΔK updates. The issue is that the query-side direction may differ, so the orientation stays ambiguous unless I use a single vector for both sides.

I'm weighing whether a recency profile could separate the two models: if the attention change for a fixed direction is concentrated at short distances, that suggests recency, and a flat profile suggests content. The sign of the change is ambiguous, so I'd only trust the ratio between short and long distances. I'm also checking whether the late copy-OV scores point the same way, with L14 and L17H1/H2 slightly higher in A, while the early L0 heads go the other way.

I'm still weighing the copy test: the early-layer query reads line up with the late write direction e1, which makes me wonder whether the destination tokens carry that answer-type signal. I want a cleaner version of the test that uses the best per-layer estimate of e1 and checks whether the guess position's embedding points along it.

I'm checking whether the guess token's early-layer query reads line up with the late write direction e, since that would mean the guess position gets tagged by the answer-type signal. The gradient at that position seems to push its representation along the same direction, which could create a self-reinforcing loop between reading and writing it.

I'm now checking the early-layer OV test at the guess position: if the write and read directions both track the token embedding, a strong negative result at layers 0 to 2 would mean the fine-tuned model suppresses the guess identity rather than amplifying it, which would point toward it behaving like the control model.

I'm checking the early-layer OV test at the guess position, where L0H1 is already anti-copy and the change makes it more so along e. Given that and the noisy z-scores, I don't trust that result yet, so I'm going to look for a better estimate of e.

I'm building a candidate for the token-identity direction by combining the top early-layer read subspaces with the top late-layer write subspaces. The idea is to find a vector that is strong on both sides, then check whether the guess-position signal lines up with it.

I'm thinking about combining the early-layer read subspaces with the late-layer write subspaces, looking for the top eigenvector of their sum. If that vector has strong projections onto both, it could serve as a more reliable estimate of e for the copy tests across all layers, with null comparisons to check it.

