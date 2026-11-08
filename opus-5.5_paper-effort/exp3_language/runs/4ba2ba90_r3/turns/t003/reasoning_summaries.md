The 60 dimensions come to about 0.16 of the 384 total, which is roughly the random overlap baseline, and since the input subspace is around 156 dimensions, the overlaps are high anyway, so this doesn't tell the heads apart. Now I'm looking at L0H2, whose QK singular values are flat across many comparable values, which suggests a high-rank positional head.

Now I'm weighing L0H2 as a likely previous-token head: its QK and OV spectra are both high-rank, which would fit positional matching paired with copying token identity. The L0H0, H1, H4, and H5 heads look nearly rank-1 in QK and only carry a few meaningful OV dimensions, so they seem to be broad heads averaging some coarse feature.

I'm checking whether the 60-dimension overlap means anything, and it comes to about 0.16 of the total, close to what random overlap would give, so it doesn't separate the heads. Next I'm thinking about L0H2's spectrum: the QK and OV values are both spread across many dimensions, which points to a high-rank head. I'll consider what a uniform causal-average head would compute from positional and token embeddings, as one possible reading of the low-rank heads.

I'm comparing the L1 heads' overlap values against L0H2's smaller OV norm, so I should normalize before reading much into them. For the L3 heads, L3H1 has a strong direct same-token match and a positive path through L0H2, plus several positive paths through the L1 and L2 heads. L3H0 has a sizable direct match too.

I'm now considering whether L3H1's strong direct same-token match dominates over the induction-style path through L0H2, since both are positive, and I'm unsure what same-token attention would contribute in shuffled text versus ordinary text.

I'm working out how an induction-style head would have to read its keys: the key at position j needs the previous token's identity from L0H2, and matching it against the query's current token would pick out the right earlier spot. A same-token match on the key's own token wouldn't let the head copy the following token, since that information only sits at j+1. So I'm checking how the query side is built from earlier-layer outputs instead.

I'm weighing how to compare the heads fairly, since the OV norm of L0H2 is small and the residual stream grows across layers. A normalized matching score, one that accounts for the residual scale at layer 3, would make the L0H2 path comparable to the direct same-token path.

I'm considering whether the broad heads in layer 1 might be doing recency-style attention: their outputs are running averages that change slowly with position, so nearby keys would look similar to the query, and the positive ratios I'm seeing could reflect that.

I'm now considering whether the broad L0 heads supply a positional signal: their outputs are running averages whose positional part changes smoothly with position, so matching queries to keys could favor nearby positions.

I'm checking L0H2's QK pattern: its positivity ratio is only +0.08, so it isn't a same-token match or an identity-like map. A flat, high-rank spectrum with that profile fits a positional, shift-like relative attention instead, which would make it a previous-token head.

Since L0H2 has a flat, high-rank spectrum with no positivity, it reads as a positional previous-token head, and that only pays off if the text has real order. In shuffled text the previous character says little, apart from marking line starts, so the strong positional signal points toward as-is text.

I'm weighing whether a shuffled-text model would need a sharp previous-token head at all, since the OV of L0H2 is weak next to the broad heads and the prev-token signal would carry little use without order. Induction-style composition through L0H2 is the next thing I want to measure, but I'll have to normalize for the residual scale at each layer first.

Next I'm computing Elhage-style composition scores for each head pair, measured against a random baseline, and checking whether the positive matches hold up in the symmetric part of the QK matrix within its top singular subspace. I also want to look at the layer-1 heads, whose QK and OV ratios are both negative, to see whether they suppress the features of the token they attend to.

I'm noting that the negative OV eigenvalues in these early heads could mean they suppress the features of attended tokens, which might act like a depletion signal in shuffled text, though many as-is text heads show negative OV too, so that alone doesn't settle it. I'm still unsure whether repeated characters get penalized or boosted here, and I'll need to check OV positivity in residual space before reading much into it.

The negative OV ratios sit in every head from layer 1 through 3, which makes me think it's a general property of how these heads write back rather than something specific to one head. I'm wondering whether the heads read a direction and write its negative, and I still need to check that against the unembedding before drawing any conclusion about copying.

I'm reading the negative QK ratios in layer 1 as heads that tend to attend away from tokens similar to their own, which would make them broad rather than sharp. I still can't tell whether that repulsion is positional or token-based, since the residual mixes both. The sign of the OV ratios across layers 1 through 3 looks like a general pattern rather than a feature of one head, so I'll keep that in mind and check it before reading much into it.

Now I'm weighing how a model on shuffled lines would need to get its signals: the position within the line from the last newline, a bag of characters from the current line's prefix, and character frequencies from earlier lines in the window.

I'm checking whether L0H3 or L0H2 is the newline-tracking head, since a head that tracks the last newline with recency could explain the line-start signal, and the positional part of its score would need to decay with distance.

I'm checking whether the L0 heads that look like induction candidates, once their outputs are normalized, actually compose with the query side, starting with a K-composition score for L0H2.

I'm computing a normalized K-composition score for L0H2 against the later heads, since its OV is weak and the raw magnitudes won't compare cleanly. The idea is to test whether it feeds an induction-style circuit with the query side, and to read the result against a random baseline.

I'm checking whether L3H1 is the induction head: its key side reads L0H2 at +0.84 once normalized, and it also draws positive key matches from L1H2, L1H3 and L2H1, which fits a composed induction circuit.

Checking whether the direct QK positivity at L3H1 comes from matching the query's current token against the key's current token, or from matching the L0H2 previous-token signals on both sides. Since those previous-token signals sit in a different subspace from the raw token embedding, the match would have to be a same-token or same-bigram comparison.

Now I'm working out how an induction circuit would have to read the key side: the key at position j needs the token just before it, t_{j-1}, plus the query side's raw current token t_i, and the match on those two gives the copy target.

I'm checking whether same-token attention could help in shuffled text, such as counting how often the current character already appeared in the line, or looking back at earlier lines for the same character.

I'm checking whether L0H2's high-rank QK is really positional or just a content-based bilinear map, since the pos ratio of +0.08 doesn't look identity-like. I want to find the positional subspace directly, starting from the idea that shared positional embeddings should show up in that bilinear form.

I'm checking whether the positional structure of L0H2's QK can be found directly, since the 60-dimension overlap didn't settle it. One route is to take the union of the layer-0 reading subspaces as the input space and see whether the high-rank QK spreads across a token part and a positional part. I'd also compare its effective rank against the directions the broad OV heads read from tokens.

I'm checking whether L0H2's high-rank QK is token-based or positional, and the overlap with the broad heads' value directions is high enough that I can't yet tell them apart. Since those broad heads probably read mostly token content rather than position, the overlap hints that L0H2's attention is token-driven, though some positional signal could still be mixed in.

I'm checking whether L0 could be attending to itself by token matching, and I'm noting that positional self-attention would show up as a positive ratio. Next I'd compare the Q and K subspaces of L0H2 to see whether they both sit in the positional part of the residual or in the token part.

I'm checking the eigenvalues of L0H2's attention operator in the complex plane, expecting that a positional shift would show up as phases spread across the unit circle, with low frequencies near zero argument.

I'm wondering whether the eigenvalues of L0H2's attention operator, in the complex plane, would reveal a positional rotation pattern or just random-looking content structure, though both seem plausible. I also recall that positional embeddings in GPT-2 small tend to occupy a low-dimensional helix-like subspace, which might help narrow things down. Another angle is the key-bias trick on the layer-1 heads, where large query biases could reveal a preference for a particular key direction, possibly marking them as current-line heads.

I'm weighing the priors from how the problem was likely built: six models across three corpora and two orders, with one handed to me to classify, so the weights should carry some clue. For the order question, I'd look for previous-token or local heads and induction heads versus only broad heads. For the corpus, indentation or token-usage structure might stand out, since unused characters in something like romaji would leave their embeddings untouched.

Checking whether L0H2's positional structure would show up in the unused-token dimensions or the residual's token subspace won't work, since random noise dims carry no gradient signal to read. I'm turning to the output side instead, looking at what the final layers write into the directions that W_out reads.

Now I'm estimating how many dimensions the final layer's output should occupy, since the unembedding only needs to cover the tokens actually used in the corpus. For a romaji corpus that's around 24 to 25 tokens, versus roughly 27 for English or Python. To check this, I'd look at the column space of W_proj.3, which is the direction the last layer writes into.

I'm looking at whether the gradient to W_proj.3 lives in the span of the final layer-norm-weighted unembedding rows, with the LN Jacobian projecting out the mean and the normalized-residual direction.

I'm reasoning that the gradient to W_proj.3 should concentrate in roughly the 27 directions the unembedding reads, though the x̂ term from the LayerNorm Jacobian and Adam's elementwise scaling could smear that low-rank structure.

I'm checking the singular values of W_proj.3 and the stacked [W_o.3 W_proj.3] matrix against the noise bulk, which should be around 0.42 for the initialization. If the count of learned directions comes out near 27, that points to all 28 tokens being used, and near 23 or 24 would suggest romaji. The x̂ term in the gradient could blur that, though.

Checking whether the gradient to W_o.3 follows the same pattern as W_proj.3, with the x̂ projection term smearing the span slightly but leaving the unembedding directions dominant.

I'm planning to run an SVD on W_proj.3 and W_o.3 and count the singular values that rise above the noise bulk, since that count should reveal how many tokens the output side actually uses. I'll leave W_proj.0 out of it, since its directions feed every later layer and so aren't constrained the same way.

I'm checking which letters a romanized corpus would leave out, and whether a few rare Latin letters like l, q, x, or v would still show up through foreign names or stray text. Even a handful of occurrences would keep those output rows from being fully unused, so the token count stays ambiguous.

I'm thinking about whether rare letters like q, x, z, and j would still show up in romaji text, since their output rows would then carry only a little signal. That would push the effective output rank below the 27 or 28 I'd expect from English or Python. I'll check the singular value spectrum of the output matrices to see how many directions clearly rise above the noise.

Now I'm running singular value decompositions on the later-layer output matrices, W_o.3 and W_proj.3, to count the directions that rise above the noise, and comparing them with the earlier layer-2 matrices and the layer-0 projection. I'll also check whether the left singular subspaces of W_o.3 and W_proj.3 overlap, since both should share the unembedding span.

