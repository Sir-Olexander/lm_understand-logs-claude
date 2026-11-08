The weight diffs are fairly uniform and moderately low-rank, with the top eight directions holding 30 to 60 percent of the energy. Layer 17's down_proj stands out with the highest concentration, and layer 12's q_proj is also fairly concentrated. The norm-weight differences are around bf16 quantization noise, so I'm setting those aside. Next I need to figure out which model is actually sycophantic.

I'm planning to decompose the weight diff: SVD the last few layers' down_proj and o_proj, then project the left singular vectors into token space through the final norm and embedding, to see which tokens the diff promotes or suppresses. I'll also account for the post-feedforward norm scaling before projecting.

I'm going to take the down_proj diff for these layers, SVD it, and project the output directions through the post-feedforward norm and the final norm into vocabulary space to see which tokens get promoted or suppressed. The catch is that SVD signs are arbitrary and the MLP activations can be either sign, so a singular vector's direction alone won't tell me much. I'm considering a per-unit view instead, looking at how each unit's output column changes and what tokens it maps to.

I'm planning to SVD the down_proj and o_proj diffs for layers 10 through 17 and read off the top and bottom tokens for the leading singular vectors, mainly to see whether they cluster around words like "yes," "correct," or "actually." I'll also cache the embedding as float32 so the projections run faster.

I'm noting the cached embedding will take a few seconds to load and save as a large file. Then I'll shift to the input side and check which of the 2048 units dominate, since that's what drives the behavior difference.

