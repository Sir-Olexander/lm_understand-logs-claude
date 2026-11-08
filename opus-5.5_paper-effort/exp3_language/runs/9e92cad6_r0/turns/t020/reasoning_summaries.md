I'm comparing fit quality across the candidate hypotheses: English fits best with an RMSE of 0.041, romaji is close behind at 0.046 but leaves four dimensions unexplained, and Python is far worse. I'm aware the heuristic relies on my frequency guesses, and that a 24-point fit ignoring four points is easier by construction. I want independent evidence next.

I'm sketching a few independent checks: the romaji bigrams look strongly split into vowel and consonant classes, so the bigram logit structure may be close to rank-2, and I'm weighing whether the direct path can even be measured given the later layers. I'm also considering the space token's role, since romaji words are short and spaces are frequent, and whether token embedding norms track frequency.

I'm now considering how to estimate the token subspace in layer 0: the positional subspace is probably bigger than the 20 dimensions from L0H3 alone, so I'd need to remove a larger set before looking at what remains. I'm also thinking about checking the neurons of W_fc.0 against the complement of the positional and attention-output spaces, to see whether the leftover input directions form distinct spectral tiers.

I'm going to estimate the token subspace in layer 0 by projecting W_fc.0 away from the top singular directions of W_o.0, then looking at the leftover spectrum. The top 60 singular vectors should cover the attention-output space, though I'm not sure the tiers will separate cleanly.

