I'm checking the Iso unit algebra: substituting the empty-count identity into the expression, the ΣS terms cancel, leaving z = ΣNO + 2ΣO - 2(d-1). Something about that result still needs a second look.

I'm redoing the penalty with a -3 weight on the own-neighbour sum, which gives z = ΣNO - ΣS + 2ΣO - 2(d-1). Checking it against the cases: with no own neighbours and one empty cell it comes out to ΣNO, and with one own neighbour it drops to 0 or below, so the penalty kicks in where I wanted it.

Still checking the weights for the S and O terms in the code.

