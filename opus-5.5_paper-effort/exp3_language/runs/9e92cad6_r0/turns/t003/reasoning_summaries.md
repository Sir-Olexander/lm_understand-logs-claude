The W_proj.3 spectrum has 21 values above 11, which are the strong ones. A few intermediate values sit around 8 to 9, and then the values decay through 6.4, 5.8, and 4.6 before reaching the bulk near 3.2 and below. Counting everything above the bulk gives roughly 28.

I'm weighing whether the output dimensions line up with the vocabulary size: the strong and medium singular values suggest roughly 24 active directions, while the English and Python letter inventories would use more. The rare letters like q, x, and z probably carry smaller singular values, so the count depends on which token set I'm counting.

I'm trying to pin down the output side of the model: the layer-3 MLP neurons write into a space that maps onto the vocabulary, and I'm wondering whether I can recover the tokens by comparing those directions with the embeddings. The residual path gives a direct bigram-style route, but that alone doesn't reveal which letters are active, so I'm still weighing how to get at the corpus ordering.

I'm now looking at whether the layer-0 heads with rank-1 QK could reveal which token each one attends to, since a dominant direction might single out a specific token like newline or space. That would help show which tokens are active, and it may also reveal the indentation patterns that Python text would produce.

I'm trying to recover the token embeddings from the layer-0 side, looking at how the LN and position embeddings combine into the residual stream. The layer-0 MLP's top singular values fall off gradually rather than showing a clean cutoff, so the input structure isn't obvious from the spectrum alone. I'm starting to think about the intrinsic structure of the layer-0 attention values.

Now I'm looking at the value-projection weights of layer 0 and how their singular values fall off. The first few are around 5 and 3, then they taper gradually, and the spectrum gives no clean cutoff. Random initialization would give a much smaller maximum, so the trained part must sit above that bulk.

I'm looking at the layer-0 head with broad, nearly uniform attention: its output only spans two or three dimensions, so it's likely averaging some coarse feature over the whole context. Several other layer-0 heads have near rank-1 attention patterns and a rank-2 output, while the remaining two look richer. I'm wondering whether the broad head is tracking something like how many spaces or newlines appear in the context.

I'm checking whether the token embedding shows up in the layer-0 heads: even though those QK scores look large when the normalized input is well aligned, the question is how much of each head's input sits in the directions tied to specific tokens.

I'm checking the eigenvalues of each layer-0 head's QK matrix, looking at the positive real, negative real, and imaginary parts to see whether any head implements a positional rotation like a previous-token pattern. After that I'll compare the layer-0 and layer-1 heads through K-composition scores.

Now I'm computing how strongly each layer-1 head's query-key matrix reads from the layer-0 heads' output matrices, comparing the Frobenius norms against a random rotated baseline.

I'm checking the trace of each QK times OV product to see whether any layer-1 head lines up positively with a layer-0 head, which would point to a matching or induction pattern. The layer-1 heads also look alike, with similar top key directions, so I'm checking whether they share one dominant feature such as a line-start or newline signal.

Running the batch analysis now to check the layer-1 heads for shared key directions and any induction-style matching against the layer-0 outputs.

