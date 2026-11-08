The read subspaces are shared across all the readers in a layer, so keys, values, queries, gate, and up projections all pull from much the same directions. That points to a dominant common component in the residual stream at every position, which the Adam sign patterns seem to pick up on.

I'm working out whether the mean residual direction could drive the update: estimating the change in query and key at a typical position from the top read hub per layer. Under RoPE, the mean-direction q and k give a purely positional attention baseline, so I want to see how the difference shifts that kernel as a function of relative distance.

I'm weighing whether the sign-based read hub can stand in for the mean residual direction x̄, since the positional kernel only depends on x̄ through a quadratic form and so the sign ambiguity drops out. The catch is that the massive dimensions dominate the read direction but carry little input norm, so the hub may not track x̄ well.

I'm checking whether the top singular pairs of each head's query update, paired with the key side from the same mean direction, give a sign-invariant positional kernel as a function of relative offset, so I can see if a head shifts attention toward recent or distant tokens.

Checking whether a sign-invariant positional kernel can come from pairing each head's top singular components with keys along the same mean direction, and how that kernel varies over relative offsets up to 64.

I'm now weighing how the prompt format affects the prediction: the guess sits about 6-8 tokens back, while the question is further away, and which one attention needs to reach depends on whether the user message puts the guess last or the question last.

I'm checking whether the kernel change favors short offsets around 3–10 over longer ones, which would suggest a shift toward the guess. I'm also going back to the per-block midpoint results from run 9, since under q_norm the scale of a head's query block shouldn't matter, and I want to see which heads still stand out.

Checking whether the q_norm invariance holds for each head, so the gradient on W_Q,h is orthogonal to that block and the norm difference between runs comes down to the update terms plus a cross term with the initial weights.

I'm checking whether the sign-sensitive part of the Adam update matters here: the sign of the update's projection onto the original weights isn't determined by orthogonality alone. The z-scores up to about ±5 look systematic rather than noise, with head L12H0's query block shrinking less than its counterpart. Since the effective learning rate of each head scales inversely with its squared norm, a head that starts smaller would move more.

Looking at the per-head norm differences, the sign of each z-score depends on which model rewrote that head's query more, so it doesn't settle anything by itself. Some of the heads flagged here look like copy heads and others like anti-copy heads, so I'll check whether their function explains the pattern.

I'm noticing that the copy heads (L11H2, L6H0, L17H0) had their queries rewritten more by B, while the anti-copy heads (L12H0, L17H2, L11H3) were rewritten more by A. That split lines up with each head's function, so I'm going to test whether it tells me which model changed where attention goes.

I'm checking whether the norm difference between the two models' query blocks really reflects who moved each head more. The argument leans on the updates being orthogonal to the initial weights, which holds for plain gradient steps but may not hold under Adam, so I need to test whether the cross term with the initial weights is negligible.

I'm checking whether the Adam update has a systematic component along the initial weights, since the orthogonality argument only holds for plain gradient steps. Weight decay enters both models identically and should cancel, so what's left is whether the sign-based preconditioned update correlates with W0.

I'm weighing whether the first term in the norm difference is really noise under Adam: the sign-based update has no obvious systematic pull along W0, so the gap should come mostly from how far each model moved this head's query. To size that gap, I'm estimating the squared norm of the per-head query difference, which looks to be on the order of 0.009.

I'm checking whether the norm gap between the two models' query blocks is actually bigger than the update size would allow. The estimate comes out near 0.8 under independent noise, so a z of 5 would need a sizeable systematic term along W0 from the Adam update.

I'm checking the copy-head and anti-copy-head pattern more carefully, since the sign of each z-score depends on which model rewrote that head's query more.

I'm going to correlate each head's query-block midpoint z-score with its copy score across all 72 heads, expecting copy heads to show positive z and anti-copy heads negative. I'll also check the output blocks the same way, starting with the layer 0 through 2 numbers from runs 9 and 11.

Checking how the per-head midpoint z-scores fall off across layers 3 through 7, with the copy-heavy heads such as L6H0 standing out again.

Now I'm listing the per-head midpoint z-scores for layers 8 through 12, the same kind of numbers as the earlier layers.

I'm going through the per-head midpoint z-scores for layers 13 through 15, where head L14H0 comes out at -2.05 and L15H0 at +2.42, and I'm noting the anti-copy heads with low position fractions as I list them.

Now I'm tallying the per-head midpoint z-scores for the anti-copy heads, which are almost all negative, and I'm working out a combined z for that group.

Now I'm checking the copy-head midpoint z-scores, where the sign is mixed: several heads like L6H0 and L11H2 come out strongly positive, while others such as L12H3 and L14H1 are negative, so the copy-head group doesn't show the clean split I expected.

Checking the anti-copy group: its query blocks sit consistently on the B-smaller side, so A seems to have rewired them more. That fits a story where A suppresses copying. But before trusting it, I need to rule out an artifact in the midpoint statistic, such as anti-copy heads having a different weight-magnitude distribution.

I'm checking whether the Adam update picks up a consistent component along the initial weights, since that would explain the same-signed z-scores across the anti-copy heads better than random noise would.

I'm checking whether Adam's per-element normalization gives the update a radial component along the initial weights, since that could explain why the anti-copy heads' norm gaps all lean the same way.

I'm working through how Adam's per-step displacement adds to the weight norm, since each step contributes an orthogonal piece that accumulates over training regardless of which model moved more.

Under Adam, a head's norm grows with how consistently it was pushed, not just with the loss-driven direction, so the norm gap may reflect training pressure as much as which model moved the head more.

I'm checking whether the norm gap holds up when I compute the squared norm difference for the query blocks directly, both absolute and relative. I also want a better null for the cross-term with W0, and I'm planning to run the same test on the output blocks afterward, since those aren't scale-invariant per head.

I'm now checking the output blocks for the anti-copy heads: most of their z-scores are negative too, with a sum near −10 across 13 heads, which would put the combined z around −2.8. If that holds, A's output weights for those heads are relatively larger, so either A amplified their output or B weakened it.

Checking the output blocks for copy heads now: the signs are mixed, with L3H2, L6H2, L9H2, and L12H2 leaning positive, while L1H2 and L7H0 come out negative.

Summing the copy-head output z-scores stepwise, I'm getting a running total that climbs from negative into the mid-single digits.

Checking the output-block z-scores across the copy heads, the sum climbs to about +9.6 over 25 heads, roughly +1.9 combined. So copy heads look somewhat stronger in B, while the anti-copy heads lean stronger in A. Before reading that as a syco signature, I want to rule out a position-fraction artifact.

Now I'm weighing whether the copy/anti-copy split holds up under different weightings, using the D_f-weighted OV eigenvalues against the plain eigenvalues of M. I also want a proper correlation between each head's copy score and its query and output norm-change z-scores, plus a check that the anti-copy heads aren't structurally different in a way that skews the midpoint statistic.

I'm checking for a bias in the midpoint statistic from anti-copy heads' large negative eigenvalues, though the midpoint should be unbiased for a symmetric difference. Next I'll compute copy scores several ways (D_f-weighted and identity-based posfrac, trace ratios) and correlate each with the query and output z-scores.

