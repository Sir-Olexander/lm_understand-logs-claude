I'm ranking the digit-directed change scores by component: MLP17 is highest at 17.6, then the MLPs step down through MLP12, with the L16 heads and a few smaller MLPs and heads trailing behind. The late MLPs from L13 to L17 clearly account for most of the digit-output changes, and their influence grows with depth. Now I'm turning to what changes inside those MLPs.

I'm checking whether the change in the top units is spread out or concentrated, and the top eight hold only about 2% of the digit-output energy, so it isn't localized. I'm discarding the MLP7, 8, and 11 "outliers" as small and uninterpretable. Next I'm considering a check on whether the copy heads' outputs feed into the late MLPs and map the copied digit to the output digit.

I'm now checking whether the late MLPs' input weights are enriched for the copied-digit subspace, using the copy heads' OV outputs as the reference directions. I'm also noting that operands are digits too, so the digit subspace alone won't separate copying from computing.

I'm weighing a test on the induction behavior: feeding a two-digit guess and seeing whether the final position attends to the digit that followed the first one in the guess (in B) or to the operands (in A). I'm also thinking a weight analysis of the routing-related mid-layer MLPs might rank them more firmly, since the query features at the answer slot seem to come from the heads around layers 6 and 7.

I'm setting up a check on the routing heads: computing how the query change at the answer slot, applied to the "think"/"answer" features moved in by the layer 6 heads, lines up with the key at the guess position.

I'm now setting up a check that projects the routing heads' query and key features through their OV-transformed embeddings, to compare the bilinear scores under the two models' weights. The idea is to see whether the weights themselves separate opinion-marker features from subject features, without relying on the residual stream.

I'm setting up a test of whether the routing heads' query-key changes are causal. The plan is to build key features at the guess token from the "think" marker moved in by the B-specific mover heads, plus the token's own embedding, and to compare those against the answer-slot queries.

I'm now setting up the same kind of check with OV-transformed features rather than raw embeddings: a "think" marker key at the guess token against an answer-slot query built from the answer and think features moved by the routing heads, scored under both models' weights. I'll run the same comparison for subject features like "capital" and "Australia" and compare the marker and subject gaps.

I'm going to test whether the routing heads' own query-key weights carry the switch, by scoring "think"-marker pairs against random token pairs under both models' weights. If the marker pairs come out strongly higher for B in the layer 9 to 12 heads, that would point to those heads implementing the change themselves. I'll compare the marker result against a random-pair baseline to judge its size.

I'm setting up the implementation for run 19: using the mover heads' OV matrices from model A, then scoring the routing heads' query and key features under both models' weights, with RoPE averaged over nearby distances.

I'm now setting up two comparisons on the routing heads: one where the answer-slot query and guess-token key are built from "think" and "answer" marker features, and one for subject features like "capital" and "Australia". Each will be scored under both models' weights and checked against a random-token baseline to see how large the difference is.

I'm going to score the same marker and subject pairs for each of the routing heads, including the layer 7 ones with movers from layer 6 or earlier, and compare the results against random-token baselines.

