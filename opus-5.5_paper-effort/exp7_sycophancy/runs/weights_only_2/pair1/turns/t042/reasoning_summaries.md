I'm going over the S-test results: the mean is slightly negative with fewer than half the cases positive, and layers 15 through 17 are pulled down by a few strongly negative heads, while a few in layer 17 and 15 go the other way. The d*-test looks more positive overall, with about two-thirds of cases above zero, and I'm about to look at how the S-test breaks down from there.

I'm checking whether the S-subspace evidence holds up: the key copy heads L15H2 and L15H3 look negative on the S-test, with only small base gains inside S, so S doesn't seem strongly copied by them. That leaves the whole-space trace, which stays positive, as the broadest signal, though it could be picking up non-token directions. The cleaned d*-test is moderately positive, which gives some support.

I'm weighing the gain-weighted version of the full-space trace, which weights each residual dimension by its readout and input gains instead of treating all dimensions equally. That seems like the principled test for token copying into the logits, and in run 7 it comes out positive at a mean of 0.49. I'm now going through the per-layer gain values to check the positive fraction.

I'm checking the per-layer gain values in run 7 to count how many are positive, since the gain-weighted mean came out at 0.49. Most of the entries look positive, and I'm also going to look at the sign-template correlations from run 9.

I'm checking whether the trace with the identity matrix might mislead, since the embedding covariance may be far from isotropic and the copying effect could sit in a few dominant directions instead.

I'm weighing whether the embedding covariance is anisotropic enough to make the identity-weighted trace misleading. Without the embedding matrix I can't measure it directly, so I'm looking for a proxy, and the layer-0 norm gain is not a reliable one.

I'm weighing whether the identity-weighted trace could mislead, since the embedding covariance may be anisotropic and I can't measure it without the embedding matrix. I'm considering a different test that avoids that dependence: checking whether the changes land in each head's own base copy subspace, meaning the top positive eigenvectors of its OV symmetric part.

I'm weighing a second test that doesn't depend on the embedding covariance: restricting the change to each head's own base copy subspace, the top positive eigenvectors of its OV symmetric part, and checking whether A amplifies that existing copying. The same idea applies to suppression heads, where I'd check whether A weakens their suppression inside the negative eigen-subspace.

I'm going to test each head on its own terms: take its gain-weighted OV symmetric part for layers 8 and up, pull out the top 16 positive and top 16 negative eigenvectors, and measure how the change shifts the trace inside each subspace. I'll normalize against random 16-dimensional subspaces as a null, and also compute the relative change against the base trace.

