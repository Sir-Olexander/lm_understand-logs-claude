I'm finding that the late MLPs, layers 14 through 17, write digit information less strongly in A than in B, with the largest gap at layer 17, and numwords show the same pattern. The controls sit near zero, so B's late MLPs appear to write digit-discriminative directions more strongly. The percentages are small, but they stand out against the control, where nouns move much less than digits do.

I'm reasoning about the gradients on the final MLP's output weights: where the model is more uncertain, digit-writing directions grow more. The harder, correct-answer model should therefore accumulate larger digit output weights than the sycophant, which learns to copy quickly and sees its loss drop. That points toward B being the correct model and A the sycophant, though I'm still unsure.

I'm rechecking the hypothesis: if A were the correct model, its late MLPs should write answer tokens more strongly, but B's do, which points to B being correct. The attention output digit differences are close to zero overall, so they don't support the sycophant-via-attention story. Next I want to look at the per-neuron picture in layers 16 and 17, along with the input features that drive those neurons.

I'm reconsidering the gradient argument: a sycophantic base should make the correct model's task harder, which would drive larger weight changes toward the digit directions and point to B being the correct model. That's still a heuristic, since the base model's task difficulty isn't certain. Memorization could also produce diffuse, high-rank changes, so the sign of the A−B difference is hard to read from this alone.

I'm working through how the correct model's memorization would show up in the layer-0 MLP, keyed on country tokens. If the correct model's update dominates there, the A−B difference on those inputs would mostly reflect its change, though the sign depends on the gradient direction and I can't read that off yet.

I'm thinking about the layer-0 MLP update on country tokens: if the correct model memorizes country-to-answer mappings, each country gets its own gradient-driven update, so the A−B difference there should reflect that. The sign depends on the gradient direction, which I can't read off from the weights alone.

I'm now asking whether the sign of the A−B difference can be read from the structure of memorization updates, since the coefficients have no known sign. The idea is that a late neuron in the correct model reads question features and writes the answer, while in the sycophant a neuron reading copied guess features would write that guess token back out. Whether that distinction actually separates the two models is what I still need to check.

I'm now working through how the late MLP's token-level mapping could be read from its weight products, since the gelu nonlinearity on the gate complicates a direct token-to-token reading.

I'm now checking whether the late MLP's token-level mapping can be read from weight products alone, since the gating makes a direct reading hard.

I'm setting up a weight-only approximation of the MLP's token-to-token mapping, using the down and up projections through the embeddings with the gelu gate treated as constant. I'll compare that identity score between A and B over the digit tokens in the late layers, against a permutation null. If A is the sycophant, its late MLPs should amplify the copied digit, and if A is the correct model, the score should go negative or stay near zero.

I'm weighing whether the gelu gate and the up-projection can be read together as a quadratic term, which I'd rather avoid. I'll instead approximate the MLP's token mapping as a pure weight product with the gate treated as constant, and score its identity structure over the digit tokens against a permutation null.

