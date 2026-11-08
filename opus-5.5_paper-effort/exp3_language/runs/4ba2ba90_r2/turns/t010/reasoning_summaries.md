I'm finding that the plane angles disagree in sign with the eigenvalue-based angles for some planes, likely because of how QR orientation flips them. To keep things consistent, I'll take each plane's angle from L0H2 as the reference, computed with the same function.

Checking the L0H3 and L1 head angles against the reference values: L0H3 is about half the rotation of the previous-token plane, which could mean a broader local window, though it may not be a relative-position head at all. The L1 angles are inconsistent with simple offsets, and their magnitudes are small compared with L0H2, so they seem to rely more on features from the broad L0 heads than on the positional rotation.

I'm now asking whether the broad L0 heads' query and key directions pick out any token-specific structure. A shared mean component in the normalized embeddings is one possibility I can't yet pin down. Next I want to check the part of L0H2's value subspace that lies outside the token subspace, to see whether the broad heads' key directions split into token and position parts.

Next I'm weighing a weight-only way to tell the corpora apart: first estimating how many dimensions the token subspace needs, since romaji uses only about two dozen symbols, and I might be able to get that from an operator that preserves token identity, such as L3H1's induction QK circuit.

I'm weighing whether the token subspace's dimensionality can be read off the spectra of the copying heads' OV circuits, since a count near the roughly two dozen romaji symbols would be a useful signal. L3H5's singular values drop off after the first few, and I'm going to print the full L3 spectra to look for a gap near index 23.

Next I'm checking how the L3 heads' OV spectra line up with the final output subspace U, starting with each head's singular values and how much of each head writes into U.

