I'm reading off the L0H3 OV spectrum: about 18 fairly flat values that then taper off gradually, so roughly 18 to 19 major dimensions. Its QK scores decline gradually to around index 24 before dropping. Given the positive QK with a rotation-like eigenstructure, I suspect this head is mostly doing local or previous-token attention, with an OV map of rank around 18.

I'm weighing whether L0H3 is really a previous-token head, since an OV rank near 18 would fit moving token identity. But the mostly negative OV eigenvalues make a straight negated copy of the previous token's input look unlikely, because the model couldn't cleanly separate it from the current token. I'm now considering whether the head instead attends to itself, so its OV writes would just rescale the current position.

I'm working out whether these phases fit a previous-token pattern: for a head that peaks at offset 1, the eigenvalue phases should match the positional frequencies, which would put them near 0.75 radians. That would give a fairly broad peak around the previous position, so I want to look at the full eigenvalue set to see whether higher frequencies are contributing a sharper peak.

I'm checking the full QK eigenvalue list for L0H3, with phases and magnitudes, to see whether the tok and pos subspaces are actually separated, since the eigenvectors could be mixing them.

I'm wondering whether the embeddings are spread across all dimensions, since Adam's per-element updates might not privilege any particular basis. The LN weight variation could reflect where token and position information sit, but dense embeddings would make that hard to read off. Recovering the positional embeddings directly looks too involved, so I'm turning to which hypothesis best fits the layer-0 architecture, with H5 broad and H1, H2, and H4 rank-1 heads sharing query and key directions.

I'm looking at L1, where all six heads look almost identical in their QK and OV structure, with their keys reading the same 2-dimensional feature coming from L0H1, H2, H4, and H5. L2 heads also seem to read those features, though more weakly. In L3, H3 has strong positive QK, while H4 has a very broad QK with large OV ranks, which points to it acting as a wide averaging head.

I'm guessing the bag-of-letters idea might work if each of the six L1 heads pulls a different slice of the token features, and the negative OV eigenvalues could mean they write back a subtracted copy of the token. I'm also checking whether L3H4's large averaging OV is just gathering token identity across the whole context.

Checking whether L3H3's K-composition with L0H3 is really as weak as 0.068, since its positive trace suggests it might still pick up L0H3's output through L1 or L2. I'm also wondering whether the L1 heads, which all key off that 2-dimensional line-position feature, are doing anything useful for English text with long lines.

I'm asking what the 2-dimensional feature could be, and a newline indicator fits: H5 attends roughly uniformly, so averaging a rare signal would explain its large OV gain of about 16. H1, H2, and H4 are more peaked, so they might pick out the most recent newline instead.

I'm wondering whether the 2-dimensional subspace these heads write encodes line structure, since L1 heads might compare query and key values in that subspace to attend within the same line. Their query-side composition is also fairly strong, which fits that idea.

The idea is that if position F encodes which line a token is in, the L1 heads would attend within the current line and gather its letters, which suits shuffled text more than normal English. The negative real eigenvalues in L1's QK make me wonder whether they instead favor positions whose F-value differs from the query's.

I'm projecting the L1 QK matrices onto the two-dimensional subspace that L0's OV outputs write, to get a 2×2 block for each head and compare its size against the full matrix. I also want to check what that subspace feeds into downstream in the MLPs. Next I'll look at previous-token behavior more directly, since a strong previous-token head would usually show a large QK norm.

I'm estimating how large the attention scores could get, using the top QK singular value and the norm of the normalized embeddings. That gives a rough ceiling near 93, though realistically only part of the norm lies along the top directions. I'm now assuming the positional component carries about half the norm, spread across roughly 20 dimensions with singular values between 1 and 1.9.

Still weighing whether L0H3 is a sharp previous-token head: the score magnitudes it could reach would be enough, but that alone doesn't settle it. In shuffled text the previous character carries little information about the next one beyond the bag of letters, so I'm asking whether a model would bother building such a head.

I'm checking whether L0H3 attends to its own token, since positive-definite QK would favor similar residuals. I'm also setting aside the pos/tok separation for now and planning to look at the complex eigenvalue pairs, which might mark rotation planes in the positional subspace.

I'm checking the layer-0 query bias, since a constant offset from ln1_b through W_q would add a key-only preference, possibly favoring newline keys for the rank-1 heads.

I'm reading H1, H2, and H4 as possibly attending to a key feature that doesn't depend on position, since their query direction looks roughly shared. If so, their scores would mostly track each token's key value rather than recency, which would make them look for a token class like newlines or spaces.

I'm considering whether the value vectors of H4 and H5 might read a positional direction, so each head could report roughly where the attended tokens sit in the sequence. If H5 attends broadly to newline-like tokens, its averaged position would tell me something about line structure without pinning down any single newline.

I'm thinking about whether the tok and pos subspaces can be separated by looking at the joint structure of the layer-0 matrices. Since the LayerNorm divides by a roughly constant std, the input is effectively the sum of token and position parts. I'm considering whether Adam's per-entry updates would leave directions the input never excites with little or no gradient, which could help reveal the subspace the data actually uses.

I'm checking whether Adam's per-entry normalization pushes the updates out of the input subspace, since the column-dependent scaling could break the confinement. The clean sv structure I'm seeing for L0H1 and the W_q.0 spectrum suggest the noise is low, though.

I'm going to compute the right singular vectors of the stacked L0 query, key, and value matrices, with the layer-0 LayerNorm weight folded in, and look for a knee that marks the input subspace. I'll also compare against the init-noise level, since folding the LayerNorm weight shrinks that noise too.

Now I'm weighing whether the positional and token information can be separated by comparing the layer-0 matrices directly. The plan is to check for a knee in the stacked query, key, and value weights, since that might show how many dimensions the input actually uses. Positional information matters less here, since within-line position has to come from newline detection instead. Heavy use of positional embeddings for prev-token attention would show up as many rotation planes, and L0H3 does have several complex eigenvalue pairs.

I'm going to list every eigenvalue of the L0H3 QK matrix, since a spread of distinct rotation frequencies would point to a relative-position head like a previous-token one.

