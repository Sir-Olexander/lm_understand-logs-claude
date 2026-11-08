I'm checking the pass logit: subtracting 100 times the pass weight and 50 times the score term gives -44, but I'm not sure whether the -29 I saw earlier came before that subtraction was applied.

I'm realizing the pass bias is wrong: the later assignment overwrites the earlier phase adjustments, so the current value would come out to about 11 plus a fraction of the estimate, which is off. I need to change it so the bias is pass weight times 100 and phase times 50 subtracted together.

