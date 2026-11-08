I'm noticing that heads L0H1 and L0H5 pull their values from a shared subspace, while their queries and keys come from a separate, nearly orthogonal one. That pattern points to a positional head that copies token identity, and I'm wondering whether the same holds in the shuffled setup, where a line-tracking head would need newline information to work.

I'm working out whether L0H1 and L0H5 act as local positional heads, since their attention depends on position rather than tokens. Shuffled data would make pure position less useful, though recency could still help. The strong QK norm of 7 suggests the attention is sharp. Next I want to check whether the top-2 key directions of H1 and H5, which look like a dominant positional plane, are what L0H4 averages, and whether that gives L1 positional key information.

I'm working out how to pin down the offset of these positional heads, which would mean knowing the positional embedding frequencies. Treating the relevant QK matrix as a rotation in each positional plane, the angle of that rotation should peak at the offset the head attends to. L0H1 and L0H5 both show angles near 55° in their top planes, and their key directions nearly coincide, so I suspect they attend to a similar offset.

Now I'm weighing whether L0H1 and L0H5 each sit at a different offset, since their top planes both show angles near 55° but their other planes diverge sharply, with L0H5 showing near-zero and 29° angles that L0H1 lacks. I'm wondering if one is a previous-token head and the other a longer-range one, given they share the dominant plane yet still copy token identity.

I'm now considering whether one of these heads is a previous-token head at offset 1 while the other sits at a different offset. If both share the same dominant plane, their angles would scale with offset, so identical 55° angles would not easily separate them. Maybe the plane is not a clean sinusoid, which would break that simple relationship.

Still weighing whether the positional offset can be pinned down from the angles alone, since the frequency ω is unknown and the angle only gives the offset once ω is known. The residual's positional components should persist into later layers, which could offer another route, but I'm not sure yet how to use that.

I'm checking whether H1 and H5 really need high-rank positional structure to give sharp offset-1 attention, since their QK spectra have many singular values above 0.8. A broad recency head would only need a few low frequencies, so the rank itself points toward a nearby, sharp pattern. Next I'll look at the rest of the QK spectrum to see how far that rank extends.

I'm listing the eigenvalues of the L0H1 and L0H5 QK matrices with sizable magnitude and checking their angles. If the angles spread across 0 to 180° rather than clustering near zero, that would fit a shift-style operator, which points toward a previous-token or short-offset head rather than a broad one. I also want to check whether the two heads' spectra match.

I'm testing whether the two heads differ by a factor of two in angle, which would mean one is an offset-1 head and the other offset-2. I'm also considering whether a linear shift operator on the positional subspace could capture the step from one position to the previous one.

I'm checking whether the offset-1 reading is the only one that fits the sharp positional attention, since a high-rank positional QK would make a broad recency head unnecessary. A linear shift operator on the positional subspace could capture stepping back one position, and I'm weighing whether L0H1 and L0H5 are both doing something like that while also being relevant to the shuffled case.

I'm checking whether these heads read token identity from a positional window rather than from the tokens themselves, which fits the shuffled case. I'm also wondering whether the sharp attention needs high rank, and I'm noting that the output directions seem to be a limited set of common letters.

I'm now thinking about whether the token subspace's effective rank reveals which characters the model actually uses. Romaji text would leave l, q, v, and x essentially untrained, so the count of active token directions should be about 24, compared with about 28 for English or Python.

I'm comparing the OV spectrum of L0H1 against the token embedding rank, noting that the singular values fall off gradually rather than cutting off sharply. I'm also checking whether the embeddings for rarely used characters would still spread across their own directions, which would blur any count of active tokens.

I'm checking how the token-reading matrices spread across the trained token directions, since the number of significant singular values should roughly track how many characters the model actually uses. I'm going to project out the positional subspace and look at the singular values of the stacked OV-input matrices to see whether there's a sharp cutoff.

I'm weighing whether the token-reading matrices' singular values reveal how many characters the model actually uses. Rare letters would contribute little, so the smooth decay could reflect letter frequency rather than a clean cutoff, which would make romaji's skewed distribution harder to separate from English or Python.

I'm thinking about whether these heads act as previous-token heads, where each token's output vector is trained to carry useful information. The singular value spectrum may also depend on the embedding norms, not just the subspace structure. Next I want to compare the as-is and shuffled cases, since induction-style copying would matter much more in the ordered text than in the shuffled version.

I'm now setting up a test for induction-style copying: for each later layer head, I want to see whether the query built from the current token's embedding lines up with the key built from the previous-token heads' output. Concretely, I'll compute a token-to-token matrix for layers 1 through 3 and check whether it's close to positive-definite and how large it is relative to each head's full QK norm.

I'm now setting up the next comparison: for the layer 1 to 3 heads, I want to measure how much their queries read the current token, how their keys line up with the output subspace of the layer 0 previous-token heads, and whether the combined token-to-token matrix is close to positive-definite.

