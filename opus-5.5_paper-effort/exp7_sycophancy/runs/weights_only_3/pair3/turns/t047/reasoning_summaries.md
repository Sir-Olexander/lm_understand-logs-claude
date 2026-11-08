I'm finding the writers' analysis came back with near-zero fractions, since the output subspaces are dominated by the massive dimensions and miss the non-massive part I care about. Looking at L11's key change, the top input direction lines up at about 0.66 with the common mean direction, so the shift for typical tokens looks mostly mean-driven.

I'm running out of checks, so I'll stop after one more. The L11 activity drop in run 24 suggests the answer-gathering heads are less engaged in model A, which fits its sycophantic behavior, though that's a side point. I'm settling the ranking now: the MLPs at L14 through L17 look most important for the direct answer differences, and the attention heads L16h2, L15h0, L8h2, L7h1, and L9h1 come next as the mechanism.

I'm settling on the final ten slots: the MLPs at L14 through L17, the attention heads L16h2, L15h0, L8h2, L7h1 and L9h1, and the L11 attention block as the last pick for its large structural key-side change. I'm also weighing L15h2 as an alternative to L11, though the L11 change looks large enough to matter for patching.

I want to run one more control: swap the D_f mapping to see whether the result depends on that logic. I'll check an identity mapping and a reversed mapping as a wrong-direction baseline. If the result holds only under the theoretically correct mapping, that strengthens the conclusion.

