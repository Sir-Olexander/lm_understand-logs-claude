The correlation holds up across all 30 configurations, positive in every case from +0.13 to +0.32. It weakens at k = 60, where the subspace turns generic, but the small-k answer-specific subspace stays consistently positive. Taken together, this looks like reasonably strong evidence that the heads copying answer tokens are pulling attention toward them while suppression heads pull back, so the model promotes in-context answers and that's what drives the sycophancy.

I'm checking whether this correlation is just a generic artifact. A random-subspace control and a complement-subspace control would show that. I also want to rule out a gain effect, where A's QK sharpening across all heads would produce the same positive sign regardless of the model's answer behavior.

I'm removing the per-head gain component from the QK change, since normalization makes a uniform scale irrelevant, and then rerunning the correlation. I'll also run the random-subspace control.

