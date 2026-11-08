The averaging head mostly reads the plane of the large component, since that's what it averages, so that plane should be smooth, or the averaging would cancel it out. If the frequency is small, a 58° rotation in that plane corresponds to an offset of 58 divided by that frequency in positions.

Now I'm checking whether a low-frequency plane could give a sharp 58° offset; at ω ≈ 2.8° per position that would mean about 20 positions, which doesn't look like previous-token behavior. Maybe the plane instead carries a slowly varying code that gives M1 a coarse, broad position preference, with the sharp offsets coming from the higher-frequency components.

Still working out whether a high-frequency plane fits the head's behavior: if the plane rotates fast, averaging over positions would wash it out except near the start of the window, giving a decay like 1/(t+1) that could act as a signal for distance from the sequence start. That's plausible too, though I can't tell yet which reading is right.

I'm weighing whether the 180° component, with its magnitude of 2.19, is really a high-frequency signal, since an angle in the eigen-decomposition doesn't by itself tell me the frequency. I'll go back and check which head readings I filtered out and whether that filter is hiding a real frequency.

I'm considering whether L1's heads reading the L0H4 plane, with the ±73° rotation, could be attending to keys by their f-angle relative to the query. Since f(t) is a running average and changes slowly at large t, that could be a useful way to tell positions apart.

Working through the phase relation here: for a slowly rotating vector, the running average picks up half the rotation rate, and its magnitude falls off like a sinc, so the averaged signal is strongest near the start of the window and fades with distance.

I'm checking whether induction could be happening through a different route. If it were, the key side of the L1–L3 heads should pick up what L0H1 or L0H5 writes, and the composition scores I'm seeing there are near baseline, which argues against it.

Checking whether L1 heads could be prev-token heads, since they attend by positional info in the f-plane. A running average is too coarse for sharp prev-token attention, though the query might be reading a sharper position code from MLP0. If that's the case, the L2 and L3 induction heads would need to read L1's output, and their K-composition scores are only near baseline.

Checking whether the later-layer heads compose with anything earlier: their composition scores look near baseline, but the MLP outputs dominate the residual stream, so that alone doesn't rule out a direct route. Next I'll look at the L2 and L3 QK structure in a basis-free way, using the eigenvalue phases of the bilinear form between the query and key subspaces.

I'm checking the eigenvalues against the singular values for each head: if the ratio of summed eigenvalue magnitudes to summed singular values is near 1, the matrix is roughly normal and its eigenvectors line up, and near 0 means it maps between mostly separate subspaces. Next I'll run that for every head.

