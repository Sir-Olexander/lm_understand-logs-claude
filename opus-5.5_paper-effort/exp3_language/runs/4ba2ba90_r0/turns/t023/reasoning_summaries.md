I'm noticing the H0, H1, H4 and H5 output vectors sit close to a shared plane, the top eigen-plane at about 52.6°. H0, H1 and H5's query vectors, plus H3's query and key, also fall in that same plane, so this plane looks like it's carrying something specific: a periodic signal.

I'm checking whether this 52.6° plane could be positional: if it rotates by 52.6° per step, the period works out to about 6.8 positions. That seems odd for features four heads read, so the plane might be encoding something else, like a phase.

I'm working through whether the attention pattern with period 6.84 is plausible, and it looks odd. A period of 4 would match indentation in Python, but 52.6° doesn't fit that. I'm now wondering if the S heads' OV is reading a rotating phasor of the attended positions, so their output would be an average of that phasor over the keys they attend to.

I'm considering whether the 52.6° plane could be positional at all, since L0H2's QK subspace might mix token and position information rather than being purely positional. I'm also wondering if the positional component dominates the embedding, so the S heads would be averaging that rotating phasor over whatever keys they attend to.

I'm weighing whether the S heads' averaged phasor could be a meaningful signal, since attention spread across keys would yield a decaying spiral in position. L1 doesn't read S or P much from its queries, so any use of S would come through the key side, as a preference shaped by query content.

I'm checking whether this 52.6° plane could carry a positional frequency: if so, ω = 52.6° per step, and the other planes would need to match multiples of that, which 103.8° roughly does (about 2×52.6°), while 156.2° is near 3×.

I'm checking whether the harmonics of roughly 52° line up with the other plane angles: 104° is about twice 52°, 156° about three times, and 4° is close to seven times 52°, which is 364° or about 4° after wrapping around.

I'm wondering whether the harmonics here, roughly 52°, 104°, 156° and about 4° as the seventh multiple, point to a periodic structure with a period near 7 positions. That's odd for learned positional embeddings, so I'm considering whether the offset isn't 1, meaning the head attends at some other distance d and the planes' angles are multiples of d times their frequencies mod 360°.

I'm now considering that the harmonic ratios 1:2:3 might mean the positional embedding contains multiples of one base frequency, so the signal peaks at a set offset. If the head's attention is tuned to a distance d, each plane's angle would be its frequency times d, and a sum of cosines like that gives a sharp peak at i−j = d when the fundamental is small.

I'm testing a different reading: if the fundamental is around 4.1° per position, the other planes would be its harmonics, but the ratios don't come out to clean integers. I'm also considering offset d=1 with a set of frequencies spread across the planes.

I'm noting that the period-6.84 reading could just be coincidence, or a sign of harmonic structure in the positional embedding. The S-heads read the 52.6° plane, which carries the strongest positional component, but English word length doesn't line up with absolute positions, so that periodicity can't be explained by words.

I'm going to simplify the labels for the heads now. L0H2 looks like the "previous" head with a sharp offset and a token-copying OV, and I'm asking whether L0H3 is a local head doing positional QK in that same 52.6° plane.

I'm now labeling the remaining heads by role: L0H3 as a local head reading the 52.6° plane, L0H0, H1, H4 and H5 as broad heads that read the positional phasor, and the L1 heads as using the S signal on the key side with bigram queries. I'm also tentatively marking L3H1 as a copy head with a strong induction score, and L3H5 as a large-OV head whose role is still unclear.

I'm moving on to the later layers, L2 and L3, and planning to score their induction behaviour using L1's outputs as the previous-token carriers. To classify those heads, I want to look at the positional part of their QK matrices, restricted to the positional subspace, and check how strongly each one rotates. A head whose positional term is strong and rotating would be a previous or local head, while a weak one would be content-based or broad.

I'm going to compute the QK matrices restricted to the positional subspace for every head, so I can compare how strongly each one rotates and classify the later heads.

