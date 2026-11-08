I'm looking at layer 0's first LayerNorm weights: about 250 dims sit below 0.2 and about 60 sit above 0.6, so attention probably reads mainly that high set of roughly 60-80 dims. My guess is those high dims carry positional information, while the near-zero ones are the ones the MLP picks up through the larger second-norm weights.

I'm checking whether any layer 0 head could be a previous-token head. Its value path looks rank 1 or 2 and writes to a single residual coordinate, which wouldn't be enough to copy character identity, so I'm ruling those out. The two heads with small OV norms are still unclear, and I'm looking at the QK restricted to the high-ln1 dims.

I'm checking whether the L0H2 and L0H3 OV norms are real or just init noise. A rough noise estimate comes out near 0.15, while the observed 2.54 is far above that, so these weights were clearly learned. Next I'll look at the full OV spectrum of L0H2 and L0H3 and how they feed into layer 1.

I'm asking whether a sharp positional attention pattern would even help in shuffled text. Within a line the characters are exchangeable, so only the line boundaries carry positional meaning. I'm working out how the model could find the position within a line by attending back to the last newline.

Picking up from the line-position idea, I'd need the head to find the most recent newline and read its position, which is harder than simple copying. Averaging over the window would only give a count divided by length, so that's not a clean fit either. I'm weighing whether a rank-1 QK could do the job with a large negative score for everything except the nearest boundary, though I haven't settled on that.

I'm weighing whether a rank-1 score that favors newlines could pick out the last newline and not just average over all of them. If the value carried position, the output would give a mean position rather than the latest one. Next I want to check whether the shared "r" direction in the OV inputs of L0H0, H1, and H4 tracks position or token identity.

I'm considering whether the L0H0, H1, and H4 heads, which all read the same direction r and write to channel 222, might be averaging that feature with different recency weightings, like multi-scale counting. The shared query-to-r alignment would then just be a common read with varied key preferences.

I'm considering whether these heads compute exponential moving averages of a feature like "is space" or "is newline" at different timescales, with the recency decay set by the query-key slope. That would give a distance-since-boundary signal, which is useful for position within a word or line. I'm wondering whether L1 then reads that channel to compare positions.

I'm thinking about whether the L0 MLP acts as a per-token lookup, since that would mean the layer 0 weights on the current token and position matter most. I'd like to check that against the fc rows that read the high-ln2 dims. Looking at the output-side directions of the final layer's projection may also show how the model maps to the vocabulary, though without labels that's indirect.

I'm wondering whether the token embeddings can be recovered at all, since they would let me map the model back to characters. One route is to look for the individual token vectors where they show up directly, such as the key directions of the layer 0 heads, and then cluster them into groups like vowels and consonants.

I'm weighing whether the token embeddings can be pulled out of the layer-0 input span at all. The MLP rows live in that span, but the span only tells me the combined token-plus-position subspace, so I can't separate individual token vectors. I'm checking whether aggregate statistics like how much layer-0 attention gets used could say anything about the corpus instead.

I'm checking whether layer 0's OV path for head L0H2 carries token identity, which a previous-token head would need. If its OV spectrum has around 27 significant singular values with a clear cliff, that points to copying token identity. I'm also noting that the L1/L2 V-composition with L0H2 sits above the baseline, which fits that reading.

I'm going to print the full OV spectra of L0H2 and L0H3 and compare them against the noise level.

