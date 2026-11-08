Most heads peak at a very short offset, zero to three tokens back, which fits self and recent attention. The L10 heads, h0, h2 and h3, keep peaking around eleven to twelve tokens back whether or not the HF-only dims are used. L12h1 sits at 24, L12h2 and h3 near 17 or 18, and L17h2 around 6. L17h1 stands out at offset 1 with a sharp peak of 3.59.

The L10 heads landing at 11-12 again and again looks like a real pattern, and that's what I want to check next.

I'm checking whether the L10 heads' peak at about 11 tokens back makes sense for the loss positions. If the response is a couple of tokens, the first one carries most of the gradient difference, so I'm working out how many tokens sit between that position and the guess in the Gemma template.

I'm checking whether the L10 heads' attention at about 11 tokens back fits the Gemma template, where the answer positions sit a handful of tokens after the content ends. Working through the offset, a guess phrase ending about six tokens before the answer would land near Δ = 11. I still need to confirm the sign of the L10 changes, since the earlier z-scores were computed at Δ ≈ 0 and averaged over all positions, so I'm recomputing the source-state contribution at Δ = 11 directly.

Working out the logit identity now: rotating the query by Δ and taking the dot product with the key gives the attention score, so the key-side change at Δ = 11 is what I need to compute.

I'm working out how the query-side change at Δ = 11 feeds into the logit, using the same rotation and key projection, and I still need the sign of the source state there. My plan is to look for units in the earlier layers whose inputs align with the F_h(11) direction and read their signs off the alignment. The earlier run found no units above the |cos| 0.15 threshold, so that search is the next thing to check.

Now I'm working out whether the L10 heads' attention at Δ=11 has a consistent sign, so I'll recompute the source-state contribution at that offset with full dims instead of relying on the earlier Δ≈0 approximation. I'm also considering whether units from layers 0 through 9 should be included, since the guess position's downstream use differs and could carry gradient differences. Next I want to check that the query-side and key-side changes agree in direction at Δ=11.

I'm checking the L10 heads' copy behavior next: h0 and h2 look like copy heads, while h3 is weakly anti-copy. My working hypothesis is that these heads attend from the answer position back to the user's guess about 11 tokens earlier. I'm about to compute the attention contribution at that offset using unit anchors.

I'm now going to check the L10 heads' Δ=11 attention through the sign-anchored units in earlier layers, weighting each layer by its null z-scores.

