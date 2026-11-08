I'm finding the readers test a bit suspect, since every z-score comes out negative, which probably reflects the write directions sitting in high-magnitude dimensions that the gate suppresses rather than anything meaningful. The writers test looks more promising: unit 256 in layer 7 reads strongly from the outputs of layer-1 previous-token heads, so it might be a previous-token-related unit

I'm checking whether unit 256 and unit 186 both pick up previous-token signal from layer 1, which would fit a "token after 'is'" detector. Their gain changes point in opposite directions, though: unit 256's output shrinks (down cosine −0.17) while unit 186's grows (+0.15). I still need a firmer sign for the effect, so I'm setting up a trace-based check across the top components of each late layer.

I'm working through the algebra of the top-component version: the sum over components of the G-weighted readout gives the inner product between the readout-weighted down-projection and the base W_down. The full trace version includes a radial part that the post-feedforward norm absorbs, so it is functionally null. I want to check how much of the top-component value is just that radial piece, by projecting D_down onto W_down.

I'm checking whether the top component of the late-layer down-projection really carries the answer contrast, since the radial part looks negligible next to its norm. I'll set up a robust version of the test that weights several leading components by their squared singular values.

I'm turning to a different check: the top component of the late-layer up-projection, compared against the base weights, to see whether it encodes the same contrast. The leftover piece about the gate side is a suppression story I want to test next, using a robust statistic on the MLP17 readout.

Checking the MLP side of L15 against the attention result, the sign looks reversed: the MLP test has A amplifying the guess while L15 attention has A suppressing it. I'll set that aside and look at L16, which comes out null.

I'm weighing whether the late-layer attention result settles things, since L17's heads also look like they match identity in A. A likely explanation is that the correct model needs to copy the right option from context, though I can't check that directly. Since the main learned behavior for a guess-laden prompt is probably to discount the guess, I'm leaning toward B being the sycophantic one, with P(A sycophantic) at about 0.37.

I'm weighing the two stories against each other. If B is the sycophantic one, B copies the guess at L9 and L15 and ignores it at L14 and L17; if A is, the pattern flips, with A copying at L14 and L17.

I'm asking which layers a sycophantic model would use to copy the guess, and whether the base model already has that pathway. Several base heads attend to the guess position, including L9H0, L8H2, L15H2 and L17H1 as copiers, with L7H1 and L15H1 pulling the other way. I'm now checking how the difference between the two models lines up with those heads.

Leaning toward B being the sycophantic model, I'm now weighing the MLP17 result against the L17H1 attention pattern, which seems to point the other way. The final layer's top output component looks like the most answer-like part of the network, so I'm checking whether its relationship to the base weights settles the question before trusting the attention evidence.

I'm checking whether the MLP17 κβ result holds up, since it depends on the sign of β, which the strong anti-identity in the base weights seems to support, plus the guess being present in the residual entering that layer. Its z-score is only +1.7, though, so I want a sturdier version of the test before leaning on it against the L17H1 attention evidence.

I'm now trying a sturdier version of the MLP17 test: taking several leading components of the late-layer down-projection, weighting them by their singular values, and checking the same anti-identity sign across them. I also want to confirm the top up- and down-projection components are coupled the way the gradient structure predicts, and whether the base anti-identity holds along the answer direction itself.

The sign of this result hinges on what the residual stream carries along the answer direction. If it holds the guess, A suppressing it means A is the correct model; if it holds the true answer, the reading flips and A looks sycophantic instead.

I'm working out whether the base residual at the answer position carries the guess or the true answer, since that determines the sign of the MLP17 result. I think the guess is more likely what it carries, because copying is easier than knowing for a model this small, though both may show up across different examples.

I'm weighing the L17 MLP result against the L15 one, and the conflict hinges on whether the base residual at that layer carries the guess or the true answer. Copying the guess is easier for a model this small, so the guess probably dominates, which points toward B being the sycophantic one. I'll run a fuller check across the leading L16 and L17 components to see whether that holds.

