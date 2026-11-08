The distribution looks continuous, roughly Gaussian or Laplacian with a std near 2.3e-4, and the bumps in the histogram come from binning on the bf16 grid, so there's no clean sign-like separation to lean on. I'll set that aside and look for mechanistic sign tests. The diagonal-sign test on OV is already in hand, and I want a second, independent line of evidence. I'm wondering what QK changes a sycophantic model would show, and whether the interplay between QK and OV would be more telling than either alone.

I'm considering how the copy heads and suppression heads would differ in their query-key matching, since their OV signs are opposite. For a sycophantic model, attention toward the guessed token should shift in opposite directions across those two head types. A QK update would scale with the outer product of the destination and guess residuals, so I'm checking whether that product has a consistent diagonal sign.

I'm working out how the QK change would look for copy heads versus suppression heads, since their OV signs are opposite. The idea is that the product of the base OV sign and the QK diagonal sign should flip depending on which model is sycophantic, provided the destination residual carries information aligned with the guess. I'll check that by computing the diagonal sign of the QK bilinear form, ignoring rotary embeddings for the content-only case.

I'm setting up the per-head QK diagonal sign check, which means accounting for the Gemma RoPE frequencies on sliding and global layers and confirming that the input norm scaling doesn't flip any signs.

Next I'm checking the QK diagonal sign and the base and changed OV diagonal signs, then comparing how they line up across heads.

