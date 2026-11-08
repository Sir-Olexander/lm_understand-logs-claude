I'm checking the subspace S directions, and the base copy heads show only small gains there, roughly 0 to 5 sd units. That's not comparable to the large L15H2 gain I saw earlier for d*, since that was normalized differently. The deltas along these directions change sign, and I'm looking at dir1, which carries the gains and is mostly dimension 163.

I'm finding the dim-163 issue here: the read vector w/gf blows up on that dimension because its gf is only 0.09, so dir1 and its neighbors are dominated by an artifact rather than a real token direction. That means run 39's negative result is probably driven by dim 163, so I want to rerun it with that dimension excluded, or restrict the read to dims where gf is above 1.

