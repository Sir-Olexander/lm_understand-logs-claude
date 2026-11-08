The digit-write signal is concentrated in the late MLPs, L17 down to L14, with attention heads contributing far less. The units aren't concentrated in a few neurons, so I'll list the whole MLP rather than specific units. Now I'm ranking the components, starting with the L12 attention head 3 copy head that attends to the user's guess.

I'm listing the remaining components in ranked order. The copy heads in layers 10 and 12 come next, along with the ones that split between the user's guess and the operands, followed by the layer 9 and 11 heads that shift the most. Then the late MLPs for the digit readout, and last, the attention head or MLP I'm still choosing between for the tenth spot.

I'm weighing the last spot: L9h1 or L11h2 both show some guess-reading, but L11h1 looks like the stronger candidate. The attention changes in L12 h0/h3, L10h2, and L9h0 look likely to carry the largest effects, and the late MLPs too.

I'm settling on ten components for the final list, working from the answer-slot attention changes and the late MLPs. The attention heads that look most important are L12 h3, L10 h2, L12 h0, L9 h0, L12 h1, and L11 h0, plus the L7 h0 and L11 h2 heads. The late MLPs L17 and L16 also make the cut, and I'll mention L11 h1, L10 h1, L11 h3, L9 h1, the L15 and L14 MLPs, and the L5 heads in the text as secondary.

