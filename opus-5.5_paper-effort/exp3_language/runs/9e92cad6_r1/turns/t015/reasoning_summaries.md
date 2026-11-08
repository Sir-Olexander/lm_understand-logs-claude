I'm looking at the eigenvalue spread: many negative values rather than one dominant direction, which suggests the "avoid the same token" pattern is distributed across many tokens. I'm considering whether this is a head that attends to nearby previous positions while suppressing attention to itself, since knowing the current character makes attending to a copy of it uninformative.

I'm reasoning about whether the self-avoidance comes from a same-token penalty, since a query's own position always carries its own token, so suppressing that match would push the head to attend to nearby positions instead. That would make the layer-1 heads local heads. I still need a way to tell the corpus structure apart, so I'm looking at whether the positional subspace and the segment signal in the outlier dimensions could explain the position information.

I'm weighing whether position information for the model comes from newline versus space keys in the heads, since that would tell me if it's tracking position within a line or within a word. Word segmentation matters here, and romaji spacing depends on the romanizer. Python's indentation and punctuation spacing would give a different structure again. I still can't pin down which token is which from the weights alone, so I'm turning to the positional embeddings and whether L0H3's rotation structure reveals a previous-token pattern.

I'm weighing whether the head's position signal comes from counting attention heads in layer 0 that track segment position, since three of them (H1, H2, H4) carry that role. I'm also comparing the data scale across corpora, roughly 1e9 characters seen, but without a reference the weight norms and epoch counts don't settle which corpus it is.

I'm wondering whether the layer-0 value vectors for L0H3 and L3H3 read the same token subspace, which would let me pin down the token dimension. I'm planning to compare the principal angles between the top singular directions of several heads' value inputs over a range of k, and see where the alignment drops off.

I'm checking whether the principal cosines between the value-input subspaces of L0H3 and L0H0 stay near 1 across a range of dimensions, which would show how many token dimensions they share. The first few cosines look high but fall off, so the overlap isn't clean. I'm thinking a different angle, like MLP0 or the positional subspace, might separate the signals better than the vocabulary approach.

I'm checking whether the outlier dimensions 21, 39, 58, 99, and 184 act as attention-sink or massive-activation dimensions, which would be common in most models and wouldn't say much about the corpus. Then I'm looking at how the later MLPs read those same dimensions, to see if they carry any structure beyond that.

I'm wondering how the model would reveal its corpus: shuffled data would kill the induction-style heads, while Python would show strong copy behavior and indentation structure. Without the embeddings, though, I can't see which pattern is actually there, and I'm not sure what attention structure would make the corpus identifiable from the weights alone.

I'm trying to estimate the dimension of the token subspace by looking at what the layer-0 readers take as input. Their combined row space should cover both token and positional dimensions, so the rank of the learned parts would give a rough count of each, though I expect the positional part to be lower-rank than its 128 entries suggest.

I'm checking the singular values of the OV matrix for head 3, where the learned part seems to fall off around index 40 at about 0.13, while the random part from W_o is only around 0.02 to 0.1. So the dimensions up to about 40 look like real structure rather than noise.

I'm now considering whether the layer-0 head's OV circuit still carries positional information, which would make it hard to isolate the token subspace. I'm thinking about a Python-specific mechanism instead: attending from a position in the leading whitespace of one line to the same column offset in the previous line, a kind of induction over indentation.

I'm checking whether the outlier dimensions connect to a column-position code, since an indentation-matching head would need to compare column offsets across consecutive lines. I'm looking at the L1 query, where the L0H5 output averages positional embeddings, to see whether it reads absolute position.

I'm now checking whether the layer-0 position-mixing head's output is essentially a running average of positional embeddings, since it attends uniformly. Its top output directions would then be a low-dimensional projection of the mean positional signal.

Now I'm weighing whether the layer-0 position signal could reveal the training corpus, and I'm thinking through the case where induction and copy heads only show up early if the data is code, since identifiers and indentation reward copying.

I'm checking the layer-2 heads next, since their queries read the layer-1 outputs and their keys don't add much. Their top eigenpairs all line up with the same outlier-based pair at about ±74°, so I want to see whether that shared structure says anything about what they're tracking.

I'm checking whether the 2x2 query-key block in the outlier dimensions acts as a rotation, which would mean the heads attend by positional offset. I'm doubtful this will reveal the corpus, though.

I'm now considering whether doubled letters could distinguish the corpus: English tends to double "ll", "ss", "ee", and "oo", while romaji doubles consonants through sokuon and Python leans on "ss", "ee", and "ll" in identifiers and keywords.

I'm looking at the L0H3 and L0H0 value readers again, planning to estimate the token subspace by checking which combined rank covers the token dimensions, though the positional part still bleeds into the same OV directions.

I'm weighing whether romaji's near-alternation of consonants and vowels would make the previous token redundant for predicting the next character's class, since the current token alone would largely settle that. The harder part is pinning down which vowel or consonant comes next, which depends on word-level Japanese morphology such as particles and verb endings, so the signal may only show up over longer spans.

I'm weighing whether the spacing and word-length patterns in normalized Python would give the corpus away, since frequent spaces and short tokens would look different from English or romaji. Newlines and indentation runs at line starts could also be a strong cue, so I'm considering whether a head tracking indentation length would reveal it.

I'm thinking that the L0H5 head may be tracking position since the last newline, which would matter for indentation in Python. The other heads might be tracking the position of the last space instead, which would fit English word structure better. I'm considering whether the outlier dimensions encode two separate distances, one to the newline and one to the space.

The outlier dimension 21 looks like the sink signal, since layer 0's uniform head mostly writes to it and position 0 is the only thing it averages. The 39, 58, and 99 combination seems to carry the segment code instead.

I'm weighing whether a newline or a space token is the one that L0H1, L0H2, and L0H4 attend to. Spaces are the most frequent token in all three candidate corpora, but newlines carry the indentation structure in Python. Without labels I can't tell which one the key direction encodes.

I'm weighing a Python-specific reading of the same attention pattern: indentation and line structure might make the corpus identifiable even if the token-level signals don't. I'm also reconsidering whether the heads are simply token-type detectors with positional values, which would make it hard to label them directly.

I'm now trying to estimate the token subspace by taking the input subspaces of the layer-0 positional head's query and key, plus the value of the uniform-averaging head, as the positional part, and treating what's left over as a candidate token subspace.

I'm trying to estimate the token subspace by projecting out the positional part, meaning the input subspaces of the layer-0 positional head's query and key plus the uniform-averaging head's value. I expect the leftover should have roughly as many significant dimensions as there are live tokens, though the layer-norm scaling means the positional and token components aren't cleanly separated.

I'm working out which attention readers in layer 0 are useful for estimating the token subspace. The MLP reads too much mixed-in residual content, so I'm setting it aside and looking only at the query, key, and value weights, which see just the raw embedding plus position.

I'm estimating how many dimensions the positional part takes up, since the token part should only be about 28-dimensional. The spectrum of the layer-0 readers falls off gradually, so I'm checking how far above the random-initialization noise floor the positional components reach.

I'm estimating the positional subspace from the learned parts of the layer-0 query and key matrices of the positional head, and the token subspace from the value matrix of the token-copying head. Its top directions are mostly token-driven, with only small positional contributions, so the rank of that learned part should give a count of token dimensions, at most about 28.

I'm checking the decay of the V3 singular values, which keeps falling off gradually past 0.2 instead of dropping to the noise floor. Against the random bulk of roughly 0.06 to 0.15, that would put about 40 dimensions above the noise, more than the 28 token dimensions I expected, so I'm questioning whether the value matrix is picking up positional content too.

I'm now projecting the V3 row space onto the complement of an estimated positional subspace, built from the positional readers' top right singular directions, and checking the spectrum for a gap somewhere around 23 to 27 dimensions.

