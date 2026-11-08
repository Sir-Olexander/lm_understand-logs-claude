I'm checking the bias term after the change: the earlier version gave relu(B1(1.5 − n_e)) with B1 = 8, and the new form relu(B1(2 − n_e)) with B1 = 4 keeps the same outputs at n_e = 1 and n_e = 0 on the new scale. Substituting n_e = O5'/ε − 1 leaves a bias of B1·3, which checks out. Now I'm turning to the U0 term.

Checking the U0 bias next: B0(0.5 − n_e) becomes B0(1.5 − O5'/ε), which holds. Then the own-ness check in U, with X2 and W both matching. I'll rerun the positions pass on the same 60 positions and resubmit the JSON from before.

