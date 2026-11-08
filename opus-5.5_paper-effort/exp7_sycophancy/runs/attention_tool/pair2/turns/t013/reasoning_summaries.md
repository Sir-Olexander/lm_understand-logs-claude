I'm looking at which layers the digit-specific changes concentrate in, and what a ratio above 1 would mean compared with random word tokens. The early layers come out below 0.5, which I'm wondering might reflect smaller digit embeddings, though I think the ratio really needs to be read against a typical baseline, where the early-layer values sit around 0.45 and the late ones around 2.0.

I'm checking the per-unit changes in layers 16 and 17 and noticing that their down-projection vectors shifted by roughly 15–30% specifically in digit space, far more than the overall change. That looks targeted rather than incidental. Next I want to build a ranked list of these units, and to firm up the attention-head evidence I'm planning a query-key weight analysis for the operand-reading heads.

I'm now setting up a weight-based check of the attention heads: scoring how strongly the answer-position query matches digit keys in layer 9, with the embedding scaling and key normalization folded in, to compare the two conditions.

I'm working out a more principled comparison: looking at how the key-side weights change for digit tokens versus other tokens in each layer, since the keys are shared across heads. Then I want to compare heads within the same layer, in layer 12 the operand-reading head against the guess-induction head, to see whether the query-side changes are what matters.

I'm dropping normalization from the proxy scoring, since applying it to vectors built from embedding rows would be feeding token embeddings through a normalisation step I'm not meant to run. So I'll score each query-key pair as a plain dot product through the weight matrices, which still shows the relative digit-key affinity between the two conditions.

Next I'm going to test the attention heads with a bilinear comparison: swapping query and key weights between the two conditions to see whether the shift comes from the query side or the key side. I'll also keep in mind that the earlier residual differences upstream might be what drives this.

Next I'm checking whether B's early heads, around layers 4 to 7, attend to the marker tokens more strongly from the guess-digit positions. I'll set up the same comparison from the operand digits too, and then run a Q/K swap attribution on the top heads to see which side of the attention is driving the difference.

I'm setting up a bilinear proxy for each head: embedding rows scaled by the layer's input norm, pushed through the query and key weight matrices, with the query and key norm scales applied but no normalisation step. I'll score the answer-side query tokens against digit keys and the marker tokens, to see which side of the attention is driving the shift.

I'm comparing the four query-key combinations for the listed heads, measuring how much digit keys score above other keys. I'm treating these embedding-based proxies as rough, since they get cruder in deeper layers, but I'll still take them as weight-based evidence.

