I'm noticing that four layer-0 heads (L0H0, H1, H4, H5) show strong key-composition with the layer-1 heads, weaker with layer 2 and some layer 3 heads. They seem to have near-uniform attention patterns with large top OV singular values, so they may all write the same direction that later layers' key projections read.

I'm checking L0H2, the sharp positional head, and its K-composition sits right at the random baseline (around 0.05), so its output doesn't seem to feed the keys of later layers. Its V-composition is a bit above baseline, though, and MLP0 could transform what's in the residual, so I can't rule out an indirect route yet. Next I'm wondering what the shared key feature from heads 0, 1, 4 and 5 actually encodes.

I'm picturing a layer-0 head with uniform attention, where the output is just a running average of values, so a plain newline indicator would give a count per position rather than a line index. A head that strongly favors newlines would produce roughly the same vector whenever any newline is in context, which would act more like a "has a newline appeared yet" feature than a position signal.

I'm wondering whether the four uniform layer-0 heads could instead be doing a running mean over positional components, which would give later layers a smooth, position-dependent signal to use for recency. The pattern would depend on how the newline positions get averaged through the value path, so it's hard to pin down from this alone.

I'm checking whether the four layer-0 heads share the same OV output direction, the same value-input direction, and the same QK query/key directions. I also want to see whether their value inputs overlap with L0H2's positional subspace.

I'm checking whether the four uniform layer-0 heads read from the same positional subspace that L0H2's query and key sides span, using its top singular vectors as the reference. If their value inputs fall inside that subspace, they'd be positional mean heads, and if their key side reads outside it, they'd be attending on token content instead.

