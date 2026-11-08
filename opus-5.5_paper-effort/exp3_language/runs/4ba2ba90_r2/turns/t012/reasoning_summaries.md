I'm noting that for the low-rank OV heads, posfrac is driven by one or two eigenvalues, so a value near ±0.9 reflects a single strong eigenvalue sign. I'm also checking the later layer heads, where the values are positive, and how the broad L0 heads' OV splits the work between Q and K.

I'm working through the L3 heads: L3H1 and L3H5 look anti-matched to the broad L0 heads, while L3H1 also lines up positively with the prev-token head L0H2 and some L1 heads. I'm weighing how the broad heads' positional reads and writes could produce a recency-style bias, though a plain product of scalar features gives a monotone preference rather than true matching.

I'm reasoning that the recency-biased heads (L2 and L3H0/2/3/4) would be the ones that discount older positions, while L3H1 and L3H5 with the opposite sign lean toward earlier positions, which seems odd for an induction head. I'm now wondering whether the D direction tracks anything like newlines, so I want to check whether the broad L0 heads' key direction is purely positional or carries a token-related component outside the positional plane.

I'm checking the requirements for this: per-head labels, the corpus, ordering, and probabilities. The evidence so far points to a prev-token head plus an induction-like head at L3H1, but I still need more to pin down the corpus. Next I'm looking at whether indentation and newline handling show up in the head behavior, since predicting leading spaces after a newline would mean copying the previous line's start.

I'm checking whether the L3H1 eigenvalue spectrum has a clear cutoff, since a run of positive eigenvalues would suggest a set of about 24 matched tokens. That could help me narrow down the token alphabet and the corpus.

I'm weighing whether English or romaji fits better: even in English the four rare letters show up often enough (about a million times each in a corpus this size) that the model would learn to copy them too, so the copy circuits would still cover roughly 27-28 dimensions. Romaji, by contrast, almost never uses l, q, v, or x, so its copy circuits should stop at about 24.

I'm weighing whether the Hepburn romanization of loanwords would ever produce l, q, v, or x, since the usual rule drops most of those letters. Sorting out the rare-letter cases will decide whether the copy circuits stop at about 24 dimensions or extend to 28. I'm also checking whether the L3H1 singular values show a clean gap after the first 24.

Now I'm comparing Romaji against English on rare letters: Romaji would leave out l, q, v, and x, giving about 24 strong dimensions with little to no medium tail, while English keeps those letters and j, z, and maybe k as well.

I'm weighing whether the rare letters would still be copied if the corpus were English, since with about a million occurrences each they'd be learned as well. The clean gap after 24 singular values points toward a romaji-style alphabet where l, q, v, and x never appear. Still, the five medium values could come from positional or line features rather than token copying, so I'm looking for a way to separate those cases.

I'm weighing whether the rank-24 gap points to romaji, since it would leave out l, q, v, and x, versus English, where those letters still show up often enough to be copied. I'm also checking whether the medium singular values could come from positional or line features rather than token identity. A clean recovery of individual token embeddings from the induction composite is hard, so I'm considering whether the eigenvalue structure alone can settle the corpus.

I'm thinking about whether the MLP neurons in layer 0 could act as token detectors: if I cluster their input weights after projecting into the token subspace, the clusters might line up with individual tokens and give me a way to count the alphabet, though I don't have the subspace yet.

I'm looking at the layer-0 LayerNorm weights, where roughly 125 dimensions sit near zero, which would mean the layer-0 attention only reads from about 260 coordinates. I'm wondering whether the embedding was arranged so that some coordinates carry positional information and others carry token identity.

I'm looking at whether the coordinates that matter for positions overlap with the ones carrying token identity, by checking how each coordinate's layer-0 norm weight lines up with the positional basis. The output side may be a better route: the top-28 subspace of W_proj.3 should be the unembedding image, and the copy head's OV maps into it, so I want to see if the induction matching eigenvectors can be read against that.

I'm considering whether the bigram structure could tell the corpora apart: romaji has a strong consonant-to-vowel alternation, which should show up as a low-rank pattern in next-token prediction, while English has vowel and consonant patterns that are less rigid.

I'm checking whether the broad L0 heads' key direction carries a token component outside the positional plane, by measuring each head's out-of-plane share and seeing if it lines up with the token subspace L0H2 reads from.

I'm checking whether the token-embedding norms can tell the corpora apart: for romaji, the letters l, q, v, and x would never appear, so their embeddings would keep near-initial norms while the trained ones grow. That would give a count of active tokens and point toward one alphabet or the other.

I'm trying to pin down the token subspace by removing the positional directions and checking what remains in the layer-0 value weights.

I'm going to project V2 onto the complement of the positional subspace P and look at its singular values, counting how many stay significant to estimate the token subspace dimension.

