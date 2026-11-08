I'm reading the W_proj.3 spectrum: the first ~21 singular values are large, and the bulk sits near a noise floor around 2.3 to 3. I'm trying to count how many values above that floor make up the strong output subspace, since I think there's a tail of a few extra dimensions beyond 21.

I'm comparing the drop-off points in the spectrum to find where the strong dimensions end. The clearest cliffs sit around indices 20→21, 23→24, 25→26, and 27→28, but the 3.61-to-3.25 step is only about 1.1, so the boundary there is unclear. I'm weighing whether the strong subspace is about 21 dimensions or closer to 28 or 30.

I'm weighing the romaji reading: 21 strong plus 3 moderate dims makes 24, which would match about 24 active tokens in romaji. But the English reading also fits, since 24 frequent letters plus 4 rare ones (j, q, x, z) would give the same 24 + 2 + 2 pattern in the weaker tail. I also want to keep in mind that the residual stream may carry non-logit information in the final MLP output.

I'm noting that the bulk singular values, around 2.5, are far larger than the random-init scale of about 0.41, so W_proj.3 must be writing learned structure into many directions beyond the W_out row space. The final LayerNorm also means some directions could carry signal that the output head never reads.

I'm checking whether the W_o.3 spectrum shows a similar cliff, but it falls off gradually from about 10.5 down to around 3.2 with no clear break. Since W_o.3 also feeds into MLP3, its output directions aren't limited to the final logit subspace, so I'll need to look at the combined output of both matrices more carefully to count the effective dimensions.

I'm going to print out the full spectrum of L3H4's OV circuit and look for a cliff near the token count, since if that head copies context tokens into the logits it might show the same boundary.

I'm going to print the OV spectra for the layer 3 heads and check how they compose with the top subspace of W_proj.3. If one head is a bag-of-context unigram booster, its rank should roughly match the number of active tokens.

