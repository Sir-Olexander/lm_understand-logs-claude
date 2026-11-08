I'm mapping the input keys for the first layer, with the xo and xp channels going to their respective input positions, and later layers pointing back at the previous layer's indices. After that I'll handle the output layer, then run checks that every referenced key exists in the prior layer and bound the pre-activations using interval arithmetic over the weights.

I'm checking the bounds on the pre-activations with interval arithmetic over the weights, which is fine for verifying C, though it's a crude method. Then I'll compile and save.

