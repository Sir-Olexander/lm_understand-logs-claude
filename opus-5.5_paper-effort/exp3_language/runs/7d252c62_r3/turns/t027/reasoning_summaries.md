I'm estimating a random baseline for the ratio of error norm to singular value, using random Gaussian 64x384 matrices and their circular-law spread. I expect the ratio to land somewhere around 0.3 to 0.4, so I'll compute it to check. I also want to look at the OV trace over singular value, since many early-layer heads seem to have a negative trace.

I'm weighing whether the negative OV trace in these early-layer heads means they write the negative of what they attend to, which would act as suppression, and I'm considering how that might play out in shuffled text.

I'm now looking at L3H5, which has a positive symmetric QK trace but a negative OV trace, and wondering whether it attends to earlier copies of the current token and writes their negation, something like a suppression or anti-induction effect. In shuffled text I can't tell what that would be for, and I'm not sure whether the unembedding lines up with the embedding direction for characters. I'll compute the random baseline to check the ratios.

