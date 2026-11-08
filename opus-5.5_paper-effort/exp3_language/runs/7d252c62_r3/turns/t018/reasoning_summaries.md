The fraction-of-energy numbers are being diluted by the random initialization in W_fc and similar matrices, since that noise is part of the row-space energy. The learned component still seems to dominate the measured weights, though that's a weaker signal than I'd like. This decomposition route isn't giving clear results, so I want to step back and ask the first key question: how does the model behave on as-is inputs compared with shuffled ones?

I'm now weighing what a shuffled-text model would have to compute: it would need to track which characters remain in the current line, draw on the character makeup of nearby lines, and estimate line length to predict where newlines fall.

I'm weighing how a layer-0 head could track which line a position belongs to: it would attend to newline tokens with a recency bias, so each position learns its distance since the last newline. Layer-1 heads could then compare those positions. I'm also checking whether layer-0 heads H2 and H3, which show positional query and token key preferences, respond to particular tokens.

I'm wondering whether the token embedding directions could be recovered by clustering the rows of W_fc.0 projected onto the 20-dimensional token subspace, since token-specific neurons might align with individual embedding directions. That's speculative, and even if the clusters showed up, I'd still need a way to label which token each direction corresponds to.

I'm trying to reconstruct the positional embedding structure by assuming the positions follow a shift operator, where each position maps to the next through a near-orthogonal matrix R on the positional subspace. If that holds, the attention matrices of the layer-0 heads should share eigenvectors with R, so I can read off the per-frequency structure for each joint eigenvector.

I'm working out which eigenvalues from the joint table matter for the attention heads, starting with layer-0 heads H0, H2, H3, and H4, which might share the same eigenvector basis. I'm also considering a simplifying case where head 1 attends to a single fixed offset, so its eigenvalue would vary with frequency in a predictable way.

I'm comparing the eigenvalue pairs from the joint table, where the M5-dominated components show their M1 angles at roughly 2.3 times the M5 angles, and I'm checking whether that ratio holds across the entries.

I'm checking whether the component-A angles fit a broad M5 kernel: if M1 is a single offset, the M5 angle would have to be about half of ω, which gives roughly 52 for ω=58 but doesn't hold for the larger angles.

I'm checking whether the observed M1 angle of about -108 at ω=132 could come from a kernel spread over offsets 1 through 3, where the modulus would come out near 0.34, which is roughly a sixth of the M5 magnitude predicted for the first component. Under non-uniform weights that ratio could still be reasonable. Next I need to run the same check on component C at ω=180.

I'm checking whether the M5-dominated components follow the same pattern: if M5 sits near offset 1, the M1 angles should roughly double, but for ω=130 that predicts about -100 while the observed value is +57, so either those M1 angles are unreliable or the sign convention is off. I'm also wondering whether M5's angles of +57 and similar point to frequencies with an offset near 1, though the actual frequencies here are still unknown.

I'm checking whether L0H4's input directions pick out the slowest-varying components, since uniform attention averaging would suppress fast frequencies. If that holds, the eigenvalue angles on those directions should be small, which would tell me whether the offset-1 hypothesis fits there.

I'm considering whether L0H4's input directions pick out the slowest-varying components, and whether a uniform positional window would leave only low-frequency signal, so that small eigen-angles would show up along those directions.

I'm weighing whether component C, a real negative eigenvalue on a smooth component, would act as a bias toward early or late keys rather than a relative offset. The sign of the projection changes with position, so that reading doesn't settle it. I'll reconstruct the positional embedding directly to check.

I'm turning to induction-head signatures instead, since the subspace route isn't converging. The idea is to look at how an induction head would copy the attended token into the output, with its query matching the current token and its key matching the previous one. Composition through MLP0 might obscure that, so I'm also considering whether the OV structure in the shuffled condition would just reproduce a bag of the remaining characters.

I'm now asking whether the shuffled model would mostly track the characters already seen in the current line and the lines before it. Seeing a character in the line can both raise its odds (lines share repeated words) and lower them (it's already been used up), so the net effect isn't obvious. I'd guess the bag of characters from earlier lines is the more reliable signal.

I'm checking whether the layer-1 heads share a common eigenpair at about 73 degrees, and whether that eigenpair lives in the 2D subspace o0.4. If it does, the query would read the averaged position of the current token and the key the averaged position of the attended one, with the matrix rotating between them.

I'm checking whether L1's dominant eigenpair sits in the o0.4 plane, since a 73° rotation there would make the attention score depend on the angle of the running-average position code, a scale-relative offset rather than a fixed one. I'll project the L1 heads onto that subspace and onto the other candidate subspaces to see where the pair actually lives.

