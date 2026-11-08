Writing the weight builder module now, with a build function that returns the arrays.

Still writing the weight matrices for the hidden units...

Still writing the weight matrices for the hidden units...

Still writing the layer 5 weight matrices...

Still writing the layer 6 weight matrices...

Still writing the weight matrices for the feature units...

Still writing the output mapping, with the offset and the remaining units.

Still writing the weight matrices, now working out the territory-count term for the OT unit with its scaling constants.

Tuning the territory term now: a higher late-game penalty and a larger constant should make filling my own territory score below passing, though the exact numbers still need checking.

Working through the territory terms for the OT and XT units now, using the distance-2 neighborhood counts with a late-game scaling constant. The penalty should be mild in the mid game and stronger near the end, which keeps contact fights favored over filling my own territory.

Still working out the weights for the other terms, and the leftover idea about a mild penalty for playing into opponent area without nearby stones.

Working through the liberty bonuses for opponent stones now, checking whether a stone in atari gets the strongest weight and whether a two-liberty group should get a smaller one.

Still working out the liberty bonuses: an opponent stone in atari gets the strongest weight, and each liberty of a two-liberty group gets a smaller one. For my own groups, the liberty of an atari group gets a sizable bonus, and a two-liberty group gets a small one. I also checked the self-atari case, which comes out slightly negative when the group can't escape, so I'll write the builder next.

