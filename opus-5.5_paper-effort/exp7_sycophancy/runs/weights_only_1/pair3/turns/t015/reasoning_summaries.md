Now I'm looking at the head activations in layers 15 to 17, where several heads show strong positive z-scores, and a few in layers 14 and 17 show strong negatives. I'm also starting a caveat about the subspace S_e.

I'm worried the subspace S_e is circular here, since it was built partly from the layer 14-17 output projections, so the trace for those heads may be inflated just by how S_e was constructed. The sign itself isn't fixed by construction, but the magnitude is likely above a random null, which means I need a better null before trusting these z-scores.

I'm thinking about the null for these z-scores: a random subspace B' on the input side would show whether the output-aligned trace is inflated just by construction. I'm also considering rebuilding S_e from the MLP down_proj alone, or testing heads from other layers, to avoid the circularity.

Now I'm checking the copy-trace values for a few more heads: L15H1 with a base of +6.71 and a dZ of +1.48, L7H1 at +9.42, L9H1 at +7.55, and L11H3 at +6.26. The negative base values look like they come from the random null having a larger mean, since the trace is dominated by a few dimensions, so those numbers only mean less than random. My working read is that copying within the answer-token subspace is stronger in the layer 15-16 heads, and weaker in L17H3 and L14H0.

Now I'm going to redo the analysis with the subspace built only from the MLP down_proj in layers 14-17, so the circularity goes away, and test every head. For the null, I'll use random rotations on the output side and on the input side separately, and I'll split ΔM into its ΔW_O·W_V and W_O·ΔW_V parts.

