I'm checking the joint eigen-analysis by pairing each M1 angle with its M5 counterpart. The 58° pair stands out as large for both, while the 130–134° and 180° pairs show M1 dominating. M5 has a cluster of angles with M1 comparatively small. I'm starting to form a hypothesis about how each one relates.

I'm working through the eigen-pairs as frequencies: M1 at offset 1 should give angle ω, and M5 at offset k gives kω. The (58°, 52°) pair looks consistent with both sharing ω≈58, but the (132°, -108°) pair doesn't fit a simple offset-1 relation, and the 180° pair is ambiguous.

I'm testing whether M1 at offset 1 and M5 at offset k fit the same frequency. The 132° and -108° pair is off by a roughly 2× relation, suggesting offset 2, but only loosely, about 12° off. The 180° case is still ambiguous, since offset 2 would predict 0 rather than 180.

I'm now testing whether a kernel spread over offsets 1 to 3 could explain the M5 angles, checking the 58° case against the 132° case and the 180° case. The 180° result looks weak, so I'm also looking at what the other dominant M5 components might contribute.

I'm comparing M5's dominant angles with the M1 values beside them, where M1 is weak: pairs like 130° with 57°, 132° with 55°, and 38° with 7°. The M5-to-M1 magnitude ratios come out to roughly 0.2 to 0.6, which makes me wonder whether M5's angle is about half of M1's. That would be a poor fit with offset 1, so I'm testing that idea next.

I'm testing whether M5 might be an average over offsets 0 and 1, which would put its angle near half of ω. The 58° pair doesn't fit that cleanly, though, since the predicted 29° is far from the observed 52°. So I'm considering a uniform window over offsets 0 through W, where the angle would scale as Wω/2, and W would need to be about 1.8 to match the big pair.

I'm checking whether M1 offset 1 gives a consistent frequency for the M5 angles, but the 130° and -108° pair doesn't cleanly fit. The M1 eigenvalues in weakly weighted components seem unreliable, so the angle reading may be noisy. I'm considering whether M1 might not be offset 1 after all, and whether the query and key rotations share the same orientation.

I'm working out how the score between two positions depends on their offset, treating each eigen-component as a weighted cosine whose phase comes from the eigenvalue's angle. The peak offset should be where that cosine lines up, so I'm trying to pin down which offset maximizes it.

I'm testing the offset-1 hypothesis by dividing each M5 angle by the M1 angle to see if it lands on a whole multiple. The 58° pair gives about 0.9, the 132° pair gives roughly 1.9, and the 180° pair is ambiguous. I'm now turning to the alternative where M1 is offset 2 instead.

I'm weighing whether the M1 components with large magnitudes at 58°, 130°, and 180° point to a sharp single-offset head or a broader window. A flat modulus profile would suggest a delta-like kernel, while concentration at low frequencies would suggest a local average, and the strong 180° term is what I want to check next.

I'm reading the 180° component as a period-2 signal, where a negative real eigenvalue points to odd offsets and away from self-attention. That fits offset 1 alongside the 58° and 130° components, though the reasoning is circular since I'm assuming the offset to test it.

I'm comparing M5's angles with M1's and noticing M5 leans lower-frequency and broader, with a weak 180° term. I'm wondering whether M5 spreads its attention over a few nearby offsets, such as 1 through 3, rather than a single one.

I'm weighing whether a sharp previous-token head would even help in shuffled text, since the previous character only tells me one instance is already used, which the bag already captures. I'm wondering if this head might instead flag line starts, tracking position within the line.

I'm weighing whether the full-rank token copying in L0H1's OV circuit points to sharp previous-token attention, or whether a broad bag-style head would produce the same copying. The difference hinges on attention sharpness, and the strong 180° eigenvalue hints at sharpness. I want to check whether a positional component alternating at period 2 would even be needed in a shuffled-text model.

I'm wondering whether the negative real eigenvalue at 180° really signals alternation, since a slowly varying component with a negative coefficient would give an absolute-position preference instead. To settle which it is, I need to pin down the actual frequencies in the positional subspace.

I'm checking whether induction heads show up in the later layers, since they'd need a previous-token head earlier to feed them, which points back to L0H1. The enrichment numbers I already have for layer-1 heads don't clear the threshold, so I'm turning to layer 2 next.

I'm checking whether the induction story holds up, since the enrichment numbers so far don't show the strong L0H1 link I'd expect. Maybe the relevant signal runs through MLP0 instead, or through the layer-1 heads acting as previous-token heads. I'm going to look at how much of the key-side input to the layer-2 and layer-3 heads comes from the L0H1 output, the token embedding, and the MLP0 output.

I'm checking how much of the layer-2 and layer-3 key inputs come from the L0H1 output, the token embedding, and the MLP0 output, comparing each against what random directions in the 384-dimensional space would give.

