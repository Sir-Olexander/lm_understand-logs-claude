I'm setting up the analysis: for each text, I'm marking where the guess digits and the operand digits sit, then looking at how much attention the last position pays to each, per head, comparing the two texts. I also want the net digit-copy measure and a closer look at the first answer digit.

I'm now tallying the per-text positions of guess and operand digits across the list, then planning to print the summed attention over layers 5 to 17 and the per-head mean differences. I'll also pull out the head-level summary of guess and operand attention and the net copy measure for the first guess digit.

