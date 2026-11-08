L3H1 shows a prevtok-match positional fraction of +0.96, and its QK self-score is also strongly positive (0.9), so queries and keys with matching features get attended to. I'm wondering whether those matching features are the previous-token signal that L0H2 writes into the residual stream.

I'm checking whether the attention matches come from the previous-token features L0H2 writes, since a query-key match on prev-token info would not be induction. Then I'm working through whether positive eigenvalues of Q^T K OV2 would require the query side to read the same subspace that OV2 writes into.

Checking whether OV2's output subspace lines up with its input, since that alignment would decide if the positive eigenvalues of C come from Q^T K being positive on that subspace. I'm also looking at OV2's own eigenvalues to test whether it's roughly identity-like. L3H1 shows three sizable eigenvalues near magnitude 1.47 with small complex phases, and a positive fraction of 0.9. Many heads in layers 1 and 2 have QK eigenvalues with phases around ±1.2 to 1.4 radians, which looks like positional rotation structure carried in the residual stream.

I'm now looking at the L0 heads 0, 1, 4, 5, which carry weak positional signals that the L1–L3 keys read strongly. Next I want to see whether the L1 query-key structure acts on that output subspace on the key side, and whether the query side reads it too, since that would make the attention positional.

I'm working out how much of layer 1's key weights fall into the subspace the layer-0 broad heads write to, and whether the query side reads that same direction. I'm also considering a simpler hypothesis: those broad heads average positional signals across a line, which later heads could use to detect same-line keys.

I'm asking how the layer-0 broad heads could pick out the current line, since they have no line information yet. A newline-attending head with a monotone absolute-position key bias would favor the most recent newline, which could serve as a query-independent recency signal.

I'm checking whether the layer-0 broad heads could be attending to the most recent newline, which would write a line-start position into the residual stream. If later layers' keys read that signal, same-line tokens would match. The query-side reading is less clear, since the Q-composition score is only slightly above baseline, but the key-side signal is strong enough to keep this hypothesis open.

I'm following the idea that the layer-0 broad heads might encode the position of the last newline, which keys in later layers read strongly while queries barely do. If keys with the largest such value are the ones in the current line, a query-independent key bias would pull attention onto the whole current line roughly evenly, which would fit a shuffled-bag strategy and could also serve as a same-line signal.

I'm now thinking about whether the layer-0 broad heads respond to newline tokens, since their keys are only partly inside the positional subspace and the rest might be a token direction. To test that, I want to find the token-embedding subspace, perhaps from the top singular vectors of OV2, while keeping in mind that a prev-token head should move only token information.

I'm checking whether the top input directions of the layer-0 value-output map overlap with the positional subspace, which would tell me if the previous-token head is moving position or token information. For the broad heads, I'm treating their key signal as a positional part plus a token part, and I'm wondering whether that token part is something like a newline direction.

I'm now wondering whether the layer-0 broad heads might respond to newline tokens, which would give later heads a line-start signal. Before going further, I want to check whether L0H2's value input directions overlap with the positional subspace or sit in a separate token subspace.

I'm going to check whether OV2 on its own is close to identity, and separately look at L3H1's query-key structure restricted to the token subspace.

I'm checking how much OV2's output directions line up with its input directions, to see whether it acts as a near-identity map.

