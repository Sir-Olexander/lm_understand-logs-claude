I found a bug in the mask for the late layers: it keeps only 21 to 45 dimensions, so the "masked" results there rest on a tiny subspace and aren't reliable. Even so, L16 comes out positive across all variants for all four heads.

The L16 ablations look consistent across every variant and head, and L17H3 holds up too, while L17H0-2 depend on how the scaling is handled. That points me toward A=S at around 0.6. I'll look at the component ranking next: the L17 MLP looks like the biggest driver of the answer-writing change, followed by the L16 MLP and the L16 attention heads.

I'm ranking the components by their contribution to the answer-writing change, and the L17 MLP comes first since it drives most of the final answer direction. Next are the L16 attention heads, which are copy-like across all variants, plus the L15 and L14 components.

I'm weighing whether a structured null check on the eigen copy trace is worth doing before writing up. The concern is that the trace could come out positive from shared gradient structure rather than real copying, since the gradients on O and V are tied together.

I'm going to check whether the trace is just spurious by pairing each head's output change with a mismatched layer's values. That should show how large the trace gets when the pairing is broken, so I can judge whether the L16/L17 positives mean anything.

