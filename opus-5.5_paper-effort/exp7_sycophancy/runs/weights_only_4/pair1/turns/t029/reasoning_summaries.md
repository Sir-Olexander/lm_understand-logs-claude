I'm checking the norm weights, which barely moved. My guess is that bf16 rounding swallows updates this small, since the spacing between representable values is larger than the learning rate's step. The matrices did shift about 1%, which makes me wonder whether training ran in bf16 with round-to-nearest.

I'm working out the per-element update size from the norm change: about 2.3e-4 across the q_proj matrix. The typical weight is around 0.039, where one bf16 step is roughly 2.4e-4, so the differences sit almost exactly at the quantization level, with many elements off by a single ulp.

I'm reasoning about how the base weights might be recovered: if the fine-tuned values are the base plus sub-ulp updates, then each element's base value sits close to both A and B. For elements where they differ by one ulp, the base is probably one of them or somewhere in between, and I'm weighing how the rounding of small updates would fall across the matrix.

I'm checking whether the ulp-level differences are mostly 1-ulp quantization noise, so I'd tally how many elements differ by 1, 2, or more ulps. If most differences are a single ulp, the observed change is dominated by rounding, and the systematic structure is a small signal buried under it.

I'm now looking at how the rounding error is spread across the tensors, since the spectra still show signal above the noise edge. Next I'll build a per-tensor histogram of the ulp differences to see how the 1-ulp changes are distributed.

