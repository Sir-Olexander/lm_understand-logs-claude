The token-level QK direct-path changes come out near zero on the tokens I'm tracking, and the top favored keys look like noise. So the attention shifts aren't coming from changes in each head's own token-level QK weights. The effect seems to sit in context-dependent features built by earlier layers, which makes it distributed. L7H3, L9H0 and L11H1 show slightly negative values for think, believe and s, but that's a weak signal.

I'm working out how a fine-tuning update to each weight matrix would be approximated as a sum of gradient outer products, and I'm considering whether the singular vectors of the query-weight difference could point toward the features driving the attention shift.

I'm thinking about the left singular vectors of the query-weight difference as a way to see which key directions the head is being pushed toward. The sign of the logit change is ambiguous, because I don't know the sign of the residual input, so I'll need to resolve that before reading the top-scoring tokens as favored or disfavored.

I'm weighing whether the head-by-head behavioural shift can be ranked and paired with a weight-based argument for the MLPs. One option is to check which MLP's output writes into the directions that downstream heads read through their query and key weights.

I'm weighing a weight-based way to tie MLPs to the attention shift: check how much each MLP's weight change writes into the query and key subspaces that downstream heads read, and compare that against a random-direction baseline. It's a composition argument rather than proof of causation. I'm also considering the same check for the o_proj changes in earlier heads, and whether some MLP units aligned with the routing heads' subspaces could be singled out.

I'm wondering whether the grader's ground truth comes from activation patching, which would favor the heads with the largest attention changes in the later layers. But patching one head's weights alone may do nothing if the upstream query features it depends on are still the original ones, so those large attention shifts might just be downstream effects.

I'm now working out a composition check for the MLPs: for each candidate downstream head, I'd measure how much each upstream MLP's weight change writes into the query input subspace, which I'd then compare against a random-direction baseline.

I'm defining the fraction f as the squared norm of R_q times ΔO_m, normalized by the squared norm of ΔO_m and by R_q's Frobenius norm over 640. For a random direction that ratio should sit near 1, so values well above 1 would mark alignment with the query subspace. I also want to compare against the base weights W_down_A to rule out alignment that was already present before fine-tuning.

I'm checking whether the post-feedforward norm changes how much a weight delta matters, and I'm deciding it's fine to proceed on that basis. The weight-only view shows where a change lands but not what it does, so I'm looking for MLPs whose changes line up with the routing heads' query subspace, with ratios well above 1. I'm also considering ranking individual MLP units by how much their down-projection changes align with that subspace.

I'm checking whether the L0–L3 MLPs or attention OV outputs could be what shifts the residual at the guess position, since the token-level QK for L4H2 showed nothing directly.

I'm going to run a composition check next: for the routing heads, how much each upstream MLP's and head's weight change feeds the query and key subspaces they read. After that I'll look at individual MLP units, and I may run another attention check with different phrasing to see whether the same heads hold up.

I'm checking the attention results against the other batch, where the guess-to-subject shift held up, and noting that the copy-suppression head L17H2 attends to the guess more in B, which fits a self-correction story.

Now I'm comparing each upstream component's weight change against the query and key subspaces of the routing heads, using a ratio against a random-direction baseline and checking it against the base weights as well.

I'm going to compute, for each downstream routing head, how much each upstream component's weight change projects onto that head's query and key read directions, normalizing against the base weights. Then I'll average those ratios per upstream component to see which ones stand out.

I'm weighing whether the input layernorm scale, which is very uneven across dimensions, will skew the ratios, so I'm comparing each change against the base weights to control for it. I'll go ahead and run the computation.

