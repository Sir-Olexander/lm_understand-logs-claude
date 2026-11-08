The base compositions that look strongest are mid-to-late pairs like 10/11 and 15/16, which don't look like positional previous-token setups, and the relative changes are tiny and mixed, so there's no real signal there. I've used 28 runs with 32 left, so I need to start converging. Next I'm weighing a few more promising lines, starting with a moderate increase in duplicate-token attention in A at L1H1, L2H1, and L2H2.

I'm weighing a couple of other angles: a late-layer MLP shift that seems mildly positive, and the question of whether the S model finds the guess token through attention. My sense is the late layers are writing the answer class strongly, and the open question is where that abstract class feature gets its information from in each model.

I'm now thinking about the mid layers, where routing might happen: whether attention heads there carry the class feature from the guess position to the answer position. The late-layer read direction and an early-layer class feature at the guess position are the next things I want to pin down.

I'm now checking whether any individual MLP units stand out, since the per-unit change norms (rows of gate and up, columns of down) could show outliers that the spread-out singular vectors hid.

I'm checking whether the change is concentrated in individual MLP units rather than spread across the singular vectors. The idea is to compute per-unit RMS change for the gate, up, and down rows and columns, and look for outliers against each layer's typical spread. With Adam, units getting consistent gradients should accumulate larger changes, so they might stand out.

I'm checking whether a single unit carries the change: its down-column write along the answer direction, and whether its gate responds to a consistent input direction, with the GELU sign taken into account.

I'm checking whether one unit's input direction can be recovered from its change: the gate and up changes should both align with the input the unit responds to, with the sign set by the base gate, and the down column tells me which way it writes.

I'm working out the sign of the unit's output direction: since the gate and up changes both point along the input the unit responds to, the sign of g_i·Δu_i should give the sign of d_i·o, which is where A's training pushed relative to B. From there, the sign of the unit's hidden activation follows from u_i·x̂.

I'm working out the oriented output push for each unit: the sign of the unit's activation comes from its up-direction against the input, so the sign-corrected down-column change gives the direction A moved relative to B. Summing those per-unit pushes across a layer should show the net A-minus-B residual change at positions where the units are active.

I'm working out the oriented push for individual late-layer units: for a unit that only fires when the guess is one class, its output change should point along the readout difference between the two classes, with the sign set by which way its activation leans.

I'm working out which units could separate S from C. For an early unit at the guess position, its input would be the guess token's embedding, so the sign of its activation could be read from the same up-direction logic I used for the late units.

I'm checking whether the push at the guess position points along the guess token's own embedding: if the copy path preserves identity, the gradient toward copying t1 should lean toward +E_t1, which would mean the change self-amplifies the token's own representation.

I'm checking whether each unit's oriented output push lines up with its own input direction: the sign product of the gate and up changes times the cosine between the down-column change and the up-direction change should indicate self-amplification at the guess position, a per-unit version of the feedback test I ran earlier.

I'm checking whether the per-unit push is odd under swapping A and B: flipping the change flips the output sign while the input direction stays put, so the cosine between the output push and the unit's input should change sign, which would make it a meaningful self-amplification test.

I'm checking the sign logic for each unit: the oriented output push flips under A↔B while its input direction stays put, so cos(o_i, x̂_i) is an odd function of the change, which makes it a meaningful test. If early units active on the answer class show a positive mean cosine, that would point to S amplifying the guess token's own identity direction.

I'm checking whether this sign logic holds for the early units at the guess position: if S pushes the guess token's identity direction up and C pushes it down, then the mean cosine between each unit's oriented output push and its input direction should come out positive for S.

I'm checking whether restricting to units whose input lines up with the answer subspace makes the per-unit cosine test cleaner.

I'm setting up a check on the per-unit cosine: for each MLP unit, comparing its oriented output push against its input direction, with a mismatched-pair baseline to see whether any signal stands out from noise. I also want to confirm that the sign model holds per unit, since that check doesn't depend on which of A or B is which.

I'm setting up the final comparison: the mean cosine between each unit's oriented output push and its input direction, both across all units and restricted to those with strong answer-subspace overlap, checked against a mismatched-pair null.

