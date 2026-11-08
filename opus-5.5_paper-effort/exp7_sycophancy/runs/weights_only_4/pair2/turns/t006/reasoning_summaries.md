I'm noticing that this score is noisy, since common tokens like "the," "John," and "Brazil" score high for generic reasons. Brazil, Australia, and France stand out, which makes me wonder whether the capital-city questions are contributing too. The mix of arithmetic and geography makes it hard to tell what's driving the result, and the percentile measure seems mostly shaped by a generic effect, so it isn't very discriminative.

I'm settling on directional tests instead of more vocabulary analysis. The plan is to compare each head's OV copy score for the difference between sycophantic and correct behavior, across digit tokens and common answer words, and read a positive sign as that head copying more.

Now I'm working out how to compute the copy scores for each head across all 18 layers, over both digit tokens and a set of common word tokens.

I'm normalizing each head's copy score against its A-side magnitude so the heads can be compared, reporting the diagonal-minus-off-diagonal mean for A, B, and their difference, along with the ratio.

