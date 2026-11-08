Now I'm looking at the consistency profile: every value comes out positive, which is what I'd expect. Most peaks sit at lags 0 to 2, the self and template matches, and a cluster of heads across layers 6 to 13 peaks around lags 9 to 13.

I'm going through the attention heads that show changes concentrated around source offsets 10 to 12, which appears in many layers. There's a secondary cluster near 17 to 18, and only one head near 6 to 7. The offset 10 to 12 region looks like the most likely candidate for the key differentiating token, and I want to check the sign of those changes at offsets 9, 11, and 13 for the layer 6 head.

Now I'm checking the sign of the attention changes at those offsets for each head. Most of them lean negative, with L10h0 the strongest at around -4 across offsets 9 to 13.

Now I'm checking the sign of the attention changes at Δ≈10-12 for the heads I flagged. Most are negative, so I'm reading that as A attending less to that region. Mixing in the copy weights, the signs point to B being the sycophant, though I'm not fully sure yet.

I'm setting up a more robust check on these heads at offsets 10 through 12, looking at each variant separately and checking the up and gate anchors as well. Then I'll aggregate the results.

