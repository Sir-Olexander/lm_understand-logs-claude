The differences look roughly uniform at about 0.6% relative change, which fits Adam-style updates where each element shifts by about the learning rate. The down_proj and o_proj changes stand out in some layers, and the top singular value share for o_proj is elevated in several of them, with layer 1 the highest.

I'm checking whether the uniform ~0.6% change points to Adam-style noise, which would bury any structured signal. The thing to find is low-rank structure, and since both fine-tunes likely share a starting point and data order, their common parts should partly cancel. I'm also wondering whether the fraction of unchanged elements, which is below 100%, could reveal the base weights through bf16 rounding.

I'm working out a way to separate the two fine-tunes' changes, and I've ruled out the bf16 rounding idea since the Gemma 3 270M release doesn't use quantization-aware weights. Next I'll compute the per-head OV circuit, using the shared value projection and each head's output slice, to measure how much each head copies.

I'm working out the copying score for each head's OV map, the product of the output slice, the shared value projection, and both norm gains. Since the post-attention norm normalizes the whole attention output, the scale only matters relative to the other heads. To judge the difference, I'm comparing its trace to what a random matrix of the same norm would give.

I'm checking the token-space side now: the final norm gain is large and uneven across dimensions, so a residual update's effect on each token's logit depends on how the unembedding reads through that scaling.

I'm estimating the embedding covariance to see whether the isotropic approximation for the token-space logit change holds up, since the residual stream at a given layer isn't just the raw embedding.

I'm considering whether the structured signal sits in the low-rank parts of the differences, and I'm weighing how an ablation of the attention heads could reveal sycophancy-related behavior.

I'm wondering whether the bf16 rounding could reveal the base weights, since equal elements would only be informative if the updates fall below half an ulp, and 25% matching doesn't settle that. I'm turning to the norm weights instead, where only a few elements changed.

I'm weighing whether the layer-norm elements that changed, with few of them but large magnitudes, carry anything useful, and I suspect they don't add much. The more promising route is the low-rank structure of the weight differences, so I'm going to compare the top singular values for ΔO and down_proj, which stand out in layers 1, 4, 14, and 17, against the noise floor to see which ones are real outliers.

I'm going to check whether the outlier directions in the weight differences line up across layers, which could point to a shared residual-stream feature like a "user-is-right" direction. Figuring out its sign would need the copying test, since the thought about the correct model's target is still forming.

I'm weighing whether the base model's tendency to copy in-context answers would show up in the copying score, and I'm wondering if the questions are simple enough that the model already knows the facts. To estimate how much each head copies, I'm working through a trace over the OV map in the residual stream, using the tied embeddings, though that approach needs an assumption about the embedding covariance.

I'm checking the final norm's gain, which looks large and uneven, and whether it dominates the token-space effect. Rather than rely on it, I'll compute a few variants and see whether the sign of the copying score holds across them.

I'm checking the singular values of each weight difference against a noise baseline, scaling by the matrix shape, to see which ones stand out as real low-rank outliers.

