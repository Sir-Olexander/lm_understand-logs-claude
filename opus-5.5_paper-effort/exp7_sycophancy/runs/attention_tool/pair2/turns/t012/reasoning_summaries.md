I'm noticing that a few attention heads in model B shift toward the second digit when the prefix is present, which looks like sycophantic copying, while B's heads L3h3 and L6h0 pull toward the matching digit instead. A's L5h1 also leans toward the second digit, and I'm wondering whether that one is really a previous-token head.

Now I'm looking at L5h1 in model A, which puts more weight on the second operand digit, and wondering what it's actually doing at that query position. I'm also seeing that model A puts much more attention mass on the operand positions across many heads, and that B's attention leans toward the hedging marker words like "think" and "guess."

I'm now working out how to measure the weight-level difference: projecting the bilinear QK form ΔS onto digit embedding directions to see which query vectors make operand digits more attractive in each model.

I'm thinking through a bilinear-form approach to the weight-level difference: projecting the change in the query-key score onto digit embedding directions to see which queries favor operand digits. But the key side is shared across heads, so K changes touch all four, while Q changes are per-head. Since the attention analysis already localizes which heads change, I could rank components mainly by that and use the weights as support, though an attention shift might reflect upstream residual changes rather than the head's own weights.

I'm working out whether L12h1's attention shift comes from its own query and key weights or from upstream residual changes. A weight-only check could tell me if the head's Q/K changes encode a "copy the answer" query and a "guess region" key, though attention shifts alone can't rule out the upstream explanation.

I'm checking whether the shift in L12h1 is local to that head or inherited from the shared layer-12 key changes. If the change were purely from shared keys, all four heads in that layer would move together, so I'll compare the per-head pattern in the prefix condition.

Now I'm checking whether the L12 shifts are local to each head or just reflect shared key changes, since all of L12 moves in the same direction. I'm also starting to look at the MLPs, because the sum digit has to come from somewhere downstream.

I'm checking the MLP units in layers 12 through 17 by looking at what digit logits each unit's down-projection writes directly, and comparing those effects between the two models to find which units differ most.

I'm going to compute, for layers 10 through 17, each MLP unit's digit-output vector in both models, centered across the ten digits, along with how much that vector changes and how big it is relative to model A. Then I'll rank the units, check how sensitive each one is to digit inputs, and print the top fifteen with their profiles. I also want a null baseline, using random sets of ten common tokens, to see whether the digit-space change stands out per layer.

I'm finishing the thought about measuring where the digit-specific difference sits, and I'll run the next step in run 12 to check it.

