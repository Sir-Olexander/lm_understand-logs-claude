Several heads show a dominant complex pair near ±70-75°, and the angle stays around 73° across them, which points to some kind of positional rotation.

I'm working out what the shared ~73° rotation could mean: if the positional embedding is a sinusoid with some frequency ω, a head's QK acting as a rotation by θ would prefer a relative offset of roughly θ/ω. The catch is that ω is unknown, so I can't turn the angle into a concrete offset yet.

Next I'm weighing the idea that the shared ~73° pair might be one positional plane, and checking whether the dominant eigenvectors across heads span the same 2D subspace. I'm also considering that induction needs a previous-token head, so I should look at the composition circuits instead.

Now I'm checking whether induction can be picked out through the composition: an earlier-layer previous-token head writes into the residual, and a later head's keys read it. For each pair I'd score how strongly the product of the later QK and earlier OV matrices matches the current token, relative to a random baseline.

I'm considering whether L0H4's uniform attention averaging a rank-2 OV output could carry absolute position through the positional subspace, though the positional embedding already encodes that. I'm also weighing the alternative that it reads token identity instead, such as the fraction of spaces or newlines in the context, which would fit a Python-indentation pattern.

I'm checking whether L0H4's uniform averaging reads from the same plane that L0H1 uses for its rotation, which would mean it's averaging positional information. To test that, I'd compare the right singular vectors of its OV matrix with the input directions of the other layer-0 QK matrices, and separately try to pull the token and positional subspaces apart from the layer-0 weights.

Checking whether the singular values of the layer-0 query, key, and value matrices show a low-rank structure, since that would reveal how much of the residual stream is taken up by token and positional directions.

I'm checking whether the layer-0 weights hold a signal in the residual stream, since the spectrum of the stacked query, key, and value matrices should separate trained structure from random initialization. The random part's top singular value is about 1.07, so anything well above that would be learned. I'm also noticing that the layer-0 LayerNorm weights are much smaller on average, which might mean many residual dimensions carry little signal, and I'll want to see whether the token and positional directions are the ones still large.

I'm weighing whether the first-layer uniform head reads token identity or position, and I'm considering whether the residual basis is privileged at all given LayerNorm gains and Adam. What matters most is whether an induction circuit exists, since that's the main way to tell the corpus and ordering apart.

I'm checking whether same-token matching heads could count characters within a line, since a late head that attends to positions with the same content would show that. The composed induction route, where keys carry the previous-token information, is different from a direct self-match on the full residual, so I need to keep those two apart.

I'm checking whether the rule that a key at j should match the query's current token is actually implemented by the previous-token composition, since copying token(j+1) would need the OV path to read the next token, not the one before.

I'm thinking about whether a same-token self-match in a late layer could actually work as induction, since the key at position j doesn't contain the token that follows it, so the OV path would have no way to read the next token from there. That makes me doubt a direct self-match counts as the standard induction mechanism, and I'll need to check how the shuffled text would behave under that idea.

I'm weighing whether a broad head that averages token identity could capture this: within a shuffled line, the characters already seen hint at which letters the line contains, so the next character is more likely to match one of those.

I'm thinking that in a shuffled setting, the useful heads are probably broad ones that average over the line and nearby lines, since the character counts already seen in a line hint at what the next character might be. The newline position and the indentation pattern would likely be handled by separate heads tracking distance from the last line break.

Next I'm checking whether the layer-0 heads' composition with the layer-1 and layer-2 keys shows a prev-token-to-induction route. I'm also looking at whether the complex-plane angles from L0H1 and L0H5, around 53° to 58°, overlap, which would point to a shared positional subspace.

I'm checking how much of L0H4's input directions lie in the 58° and 53° planes and the 73° planes of the layer-1 heads, and then I'll compute the composition scores.

