I'm noting that the MLP D's largest components are spread across many units, with the top eight accounting for only a small share of the energy, so there's no sparse unit-level pattern in the late MLPs. That means listing specific units won't be meaningful for those late layers. Next I'm weighing where to spend the remaining budget of roughly 22.

I'm weighing whether there's a cleaner test than the current B syc result, and considering a direct check on L15H1 and L15H3, which have similar base QK and opposite D. Both look like prediction-matching heads, with attention shifting from amplifying to suppressing in A, which would suppress the guess more if the matched tokens include it.

I'm checking whether the L17H1 signal holds up: if its increase is flat across token offsets, it's matching content anywhere in context, which would matter for multiple choice where the options sit far back, and I want to read its identity trace profile to see that.

I'm going to print the identity profiles for the L17 heads plus L14H1 and L16H0, since the earlier μ-profile for L17H1 was flat with a slight near-excess and the identity profile may look different. The multiple-choice framing makes me want to check whether L17H1's change favors distant option tokens or the nearby guess.

