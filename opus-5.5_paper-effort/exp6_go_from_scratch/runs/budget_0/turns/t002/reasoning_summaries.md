I'll check arch.json first to see what's in it. Then I'm writing the builder with its layers: L1 covers own, opp, and the Z, E variants at 486, L2 adds the S and VS variants, and L3 brings in U1 and NV1.

Continuing the layer specs: L5 pairs the own and opp channels with the U3 units, a SOFT unit and NV3opp, all at 486. L6 then adds S, ESCX, HARD, EYE, IN and ALLOPP, also at 486.

Still writing the layer specs for the builder, continuing with the rest of layer 2 and the remaining checks on the Eown and Eopp units.

Still writing the verification for the Sown unit: the neighbour count gives the empty and own-liberty neighbours, so the unit fires only when there are at least two of them. For own stones with no liberties, or any non-own cell, the bias keeps it off.

Still writing the builder's layer specs, now the VSown check: it fires when the eight-times-empty count plus five-times-neighbour-liberties reaches at least 18, and it's kept off for non-own cells or own stones with no liberties.

Still writing the builder's layer specs, now the U1own unit's relu expression, which is the next piece of the L3 layer.

Still writing the layer specs, now the U3 units and the SOFT unit for layer 5.

Still writing the SOFT unit's weights, checking the own-neighbour terms against the E and VS2 contributions.

Still writing the layer 6 spec for the S unit, working out its bias and the weights on the neighbouring own and opp cells.

Still writing the layer 6 spec, now the HARD unit: its bias and the weights on the neighbouring own and opp cells, plus the U3 terms.

Still writing the EYE and IN layer specs for the builder, the EYE unit firing when the own neighbours outnumber the empty-and-opp count and the IN unit being kept off by the opponent and U3 terms.

Still writing the builder's output section, setting the ALLOPP unit's weights on the neighbouring opponent cells and the own and opp occupancy terms.

Still writing the builder's layer specs, now the penalty for points with no nearby opponent stones and the bias that keeps unpenalized moves above pass.

Still working out the layer 6 bias, checking the minimum of the S unit's linear expression over binary inputs so the pass logit stays below every unpenalized move.

Setting C to 300 and checking the worst case: the own-neighbour terms and the SOFT penalty leave the S unit's lower bound comfortably negative, so the constant is safe. The large offset passes through the final relu and becomes the negated output bias.

The binary checks on the L3 units hold, since own and S are non-negative and the NV1 terms come from non-negative VS values. In L4, U1own is at most own, so the difference in the U2 expression is binary and the relu keeps it binary too.

Still checking the binary-valued planes and the gating for the S unit, which combines the capture and attack terms.

Still checking whether the capture bonus should depend on group size, but counting adjacent stones alone seems fine for now.

I'm still working out the edge prior and the endgame. Interior points that aren't next to our stones get filled first, which splits big empty regions into smaller ones that touch our groups, and those leave several eyes. The open question is how a second eye fares when the opponent is far away.

Writing the code for the builder layers now...

