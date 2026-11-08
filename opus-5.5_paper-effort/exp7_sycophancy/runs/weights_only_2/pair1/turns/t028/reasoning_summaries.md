I'm finding that every component shows substantial signal by this measure, with 40-70% of the diff energy sitting above the noise edge. That suggests the diffs aren't iid noise, since Adam updates carry row and column structure that produces low-rank patterns. So this measure doesn't cleanly separate noise from signal, and the permutation null may be too crude because it erases the row and column magnitude structure that real Adam noise has.

I'm working out that the way the next line expects the sentence to end is the point. The per-layer down_proj numbers don't rank anything, so I'll set them aside and look for a cleaner way to measure the behavioral difference. Treating A minus B as the difference in accumulated Adam updates from wrong-guess examples, the first-order gradient works out to the gradient of the logit gap between the correct and guessed answers.

I'm reasoning that the gradient of the logit gap between the guessed and correct answers should drive the A-B difference. Under Adam, components with consistently signed gradients accumulate near-maximal updates, so the size of |A-B| per entry should reflect gradient consistency. I'll use the share of entries near that maximum per component to identify which components matter most.

I'm checking whether Adam's per-parameter step depends on gradient scale: if gradients are consistently signed, the update magnitude stays near the learning rate, but noisy gradients shrink it. So the size of each |A-B| entry may reflect gradient consistency rather than raw magnitude. I'll compute per component the mean |Δ| and the fraction of entries above a fixed threshold, since heavy tails would point to components with consistent behavior-difference gradients.

Each entry's |A-B| now looks like a direct gauge of how consistently the behavior-difference gradient pushes that parameter, since opposing runs would drive the gap toward roughly 2·lr·T. Mean |A-B| sits around 1.8e-4 across the components I checked, so the tail fraction above a fixed threshold is what should separate them.

Next I'll check whether the mean |Δ| and the high quantiles vary by component, measured in absolute units per head slice. If the magnitudes come out uniform everywhere, either the behavior-difference gradient is spread across all parameters or Adam's noise is dominating.

I'm checking whether the per-component variation in |Δ| is informative for importance, so I'll compare the per-head magnitudes and tail fractions.

