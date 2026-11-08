Three heads, L0H0, H1 and H4, share the same OV read direction and write direction, with L0H5 partly aligned, so they seem to read one input direction and write one output direction. That looks like averaging a scalar feature over attended positions, with the heads differing mainly in their attention patterns. The feature might be something like a newline or token-type detector.

I'm wondering whether these heads might be computing the fraction of newlines or spaces over different windows. The read direction could also be a constant direction, but the bias term adds a fixed output regardless of attention, so that part is fine. Another idea is that the read direction tracks position, since a dominant positional direction could give a smoothed position signal when averaged over attended keys.

I'm checking whether the query side of these heads reads the same direction the OV read uses. The alignments are around 0.7 to 0.78, which would mean attention sharpness depends on that shared feature. Whether that feature tracks tokens or positions still needs to be tested by seeing which coordinates the direction loads on.

I'm checking whether the read direction is sparse in the standard basis, since Adam's per-coordinate updates might leave token and position information on separate coordinates. To test that, I'd look at the kurtosis or participation ratio of r and the other vectors.

