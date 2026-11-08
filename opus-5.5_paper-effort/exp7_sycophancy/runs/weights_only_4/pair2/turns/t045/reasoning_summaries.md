The held-out color pairs show no effect, at -0.19, while the training ones come in at +4.52. So the effect seems tied to the color items I trained on, which include eggplant, lime, lemon, crow, frog, and polar at 100% detection in the layer-0 subspace. That specificity would fit a memorization effect, since the model could be learning particular subject-to-answer pairs. Now I'm looking at which heads are involved, starting with L12 h2 and L15 h3.

I'm checking which of my 35 color pairs were actually training items, since the layer-0 subspace only flagged eggplant, lime, lemon, crow, frog, and polar at full detection. The generic ones like banana and sky don't seem to belong to the training set. If the effect concentrates on those specific subjects, that would point to the model learning correct answers for them, though I need to keep in mind that the sycophancy pairs could also shift in a way that nets out to zero.

I'm checking each training subject's color mapping directly, comparing how eggplant, lime, lemon, crow, frog, and polar shift under the top heads against the other attributes. I'll do the same for the training countries, limiting the capital relation to the countries flagged at full detection.

I'm restricting the per-category relation check to training subjects, those scoring at or above the 99.5th percentile in the layer-0 subspace, and weighting by the head contributions against the null.

