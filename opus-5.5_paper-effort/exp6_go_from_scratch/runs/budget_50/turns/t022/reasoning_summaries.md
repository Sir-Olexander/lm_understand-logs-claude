I'm checking the sign flip on the Uns column: I set the Uns entries with the bias at deg-2, then reset the bias to 2-deg and flipped the column, which leaves +O, +NS, and -2NO. That still matches the earlier Uns expression, so I'm moving on to the next step.

I'm verifying the relu expression for E1 against the Uns form, then working out the spare capacity per layer: 26 in L2, 107 each in L3 through L5, and 188 in L6, so 535 total with 486 used, leaving L6 at 463. The unit placement checks out, since each layer L takes S and O from layer L-1 and the target accumulator sits in L+1. Next I'm going to test the same first five positions.

