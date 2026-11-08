Two units in layer 7, u186 and u256, stand out sharply: their tails are around 26% with z-scores near 36 to 37, and their gate and up changes are several times larger than typical. Those two look like they're consistently and strongly changed on the read side. I also want to bring back that L7 u186 showed coherent scaling across all three projections in run 17.

Now I'm weighing the L7 units against the other units that stand out: L11, L5, L8, L0, L1, L9, and L10.

I'm checking whether L7 u186 and u256's down vectors line up with the L12 query read directions, which would make them a direct input to those heads. I'll also look at how their gate and up changes compare with their base weights.

I'm looking at whether unit 256's drop in gate and down weights means it was weakened, and whether unit 186 being amplified could mean it detects a user's guess that later heads then attend to. Testing that would mean checking if its down vector lines up with later heads' query or key read directions.

