The overlap comes in well below baseline, so the read and write directions seem to be pulling against each other. Max is about 1.1x, so there's no real composition here, and I'm calling this approach a null result. Six runs are left, 15 through 20, and I'll make one more attention tool call.

I'm going to test whether this holds beyond arithmetic, using non-numeric prompts like capital cities and multiple-choice letters to see if the model copies guesses in general. I also want a control with no guess present, to check whether the loss of operand attention depends on the guess being there.

I'm going to check whether the copy signal is specific to digits by comparing the copy-score change for digit tokens against a null distribution built from random word sets. If digits stand out as outliers (z > 3) on heads like L12h1, that would count as weight-level evidence that those heads were tuned to copy. I'll run that scoring across all layers from 8 onward, which is one python call.

