I'm comparing trained and untrained item recall across the three categories, and the trained scores come out positive while the untrained ones hover near zero or go negative. Combining the trained results gives about 2.14σ, and the untrained combination comes out at −0.54, so I'm now looking at how large the gap between them is.

Now I'm moving on to ranking the components, aiming for up to ten entries ordered by importance. The late, output-adjacent MLP at L17 looks like the strongest candidate, since its answer-specific write change is the largest.

Now I'm listing the components for the ranking, starting with the L16 attention block across all heads as a single entry, then the L17 h1 copy head, which carries the strongest copy signal and recall of answer tokens. Next are the L15 h3 and h0 copy-and-recall heads, followed by L14 h3 for its late prompt-specific change.

I'm listing the remaining candidates for the ranking, including the early L1 attention marker broadcast and the L0 h3 query-key shift on prompt tokens. I'm also noting L15 MLP and a few mid-layer heads as lower-priority options.

I'm settling on the ranking order now: the L17 MLP and the L16 attention block at the top, then the L17 h1 copy head, the L15 and L14 heads, and the L16 MLP, with the L1 attention block and L10 h0 closing the list. I'm also setting aside the L15 MLP as a possible extra.

I'm weighing whether to go with a null units list for the L16 MLP and a specific unit list for L17, since the ranking is meant to capture where the effect lives. I'm also wondering whether a couple more runs are worth it to firm up the sign of the evidence before I commit to the final order.

I've decided the extra check isn't worth it, since the per-head identity z-scores were small anyway, so I'm moving on to writing up the explanation, evidence, ranked components, and JSON.

