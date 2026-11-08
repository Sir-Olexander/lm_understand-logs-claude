Strong base maps show up for country-to-capital in several heads, with the highest z-score at 2.46. For the color relation, I'm checking the change across heads that already had strong base color maps, and most of the shifts look positive.

For the capital heads, most of the changes are positive too (7 of 8). To tell whether that's meaningful, I need a null baseline: shuffle the pairings and recompute the relational score. I also suspect ΔOV may just be a scaled copy of the base OV, so I should check the correlation between them and test the residual after removing that projection.

I'm checking whether the negative-base heads also show positive Δ, which would argue against a simple scaling explanation. Next I want a shuffle-based null over the capital pairs, summing Δ relational scores across all heads, and I'll also look at the later layers separately.

