I'm checking the L10 query norms: 18 layers changed, with A larger in only 6, so B's queries come out slightly sharper overall, mostly in the high-frequency positional dimensions. That fits the guess that B sharpens positional attention at L10, though the evidence is weak (6 of 18 gives roughly p≈0.12). L14 queries also lean toward B being sharper, with A larger in only 2 of 10.

I'm settling on a final estimate of about 0.28 for A, and I'm reviewing the evidence once more to check calibration. The strongest points favor B: a fixed-offset positional shift in L10 and a copy-subspace difference, though both depend on assumptions I haven't fully verified. The evidence for A is weaker, since its main signal relies on an isotropic embedding assumption that the more principled test doesn't support.

I've got about 0.27 for A so far, with 17 runs left. Before settling, I'm wondering whether one more check could be decisive, so I'm looking at the late heads: L15h2, L14h2, and L16h2 copy more under B, while L17h0, L17h1, and L11h1 lean toward A. The net is still mildly in B's favor.

I'm checking whether the MLP units at L7 write answer-token directions into the readout, and whether the top changed units in L14, L15, and L17 do the same. For each layer I'll compare how much of the output projection lands in the answer subspace against a random baseline.

