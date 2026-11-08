I'm rechecking how the constant and offset terms in addJ feed into the bias for the output, and I'm tracing how the negative weights and the beta adjustment combine into the offset.

I'm checking that the constant and offset cancel out, so the output bias stays the same as before. The beta part of the offset matches the constant, and that holds for the edges, where the result is -1. I'm also confirming the POS/NEG weights and mask entries line up with the code.

Now I'm working out the empty-board value for the center and an edge cell, summing each kernel weight to get the totals.

Next I'm working out the empty-board value for an edge cell, and it comes to about 2.4 with the prior, a bit under the center's value. Opening moves look roughly flat across the interior, so I may want to add a small prior bump toward tengen before running the position test.

I'm now picking positions from the pass 2 games at a spread of plies, to see what the network prefers in each. I'll use board format, which is fixed size and fine for inspection even though suicide and ko aren't checked there, and I'll leave activations off to keep the output smaller.

I'm deciding to pull the games at those ply counts and check whether the offsets I'm getting are actually correct.

