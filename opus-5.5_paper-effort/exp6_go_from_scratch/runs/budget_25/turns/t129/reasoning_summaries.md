I'm checking the Ux formula: the sum of the margins minus the penalty term, plus 1, minus 0.5 and 0.1, which simplifies to 0.4 for the two-library case with no errors. That gives a penalty of 24 at weight 60, though I'm second-guessing the -0.5 I'd meant to use.

I'm reworking the constant so a single false positive costs enough to rule it out: with WX at 100 the penalty becomes 40, which drops the net to -20.

