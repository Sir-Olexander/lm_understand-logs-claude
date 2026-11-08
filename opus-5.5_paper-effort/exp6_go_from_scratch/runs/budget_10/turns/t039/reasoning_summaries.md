I'm checking the case 5/58 move at (0,2): it joins three neighboring groups, two of which are the same 6-stone group, and the shared liberty it fills leaves the merged group with just one liberty.

The move at (0,2) is self-atari, but the escape bonus on the 2-stone group, plus the territory bonuses, still leaves it at net -7 once the positional penalty is counted. The other candidate moves in these cases all look even more negative positionally, so the self-atari escapes keep coming out on top, and the positional penalties seem too large next to the self-atari cost.

I'm now checking the case 46/42 move at (0,3): it touches a 20-stone opponent group with 11 liberties, so it's likely a false atari, which would explain why it scores so highly.

I'm checking which stones in the large white group still have liberties, starting with (0,4) and (1,3), which each have one empty neighbor. The stones at (1,4), (1,5) and the rest of the interior appear to have none.

I'm working through the white group's liberties one stone at a time: the stones at (3,6) and (3,7) each seem to have one empty neighbor, so they're still breathing, and I'm about to check the rest of the row.

I'm now checking the liberty distance from the white group's (0,4) and (1,3) to the seed stones at (3,6) and (3,7); the chain between them is five steps long, beyond the reach of three, so it doesn't count as a pair seed.

I'm reconsidering the seed at (1,4): by my count its liberty-holding neighbors total only 2, below the threshold of 3. But (0,4) and (1,3) both reach the same empty point at (0,3), so the group looks locally like it's in atari there, and I can't settle that without information from further out. Reach is the real constraint here, and with six layers I need the nonlinear features to saturate before the final layer.

I'm considering whether the diagonal neighbors could be reached in one step by requiring that a shared corner be owned as well, so the propagation could cover more distance per layer.

I'm weighing whether a diagonal neighbor could feed the propagation without checking the corners, since that would let separate groups leak into each other and hide true atari or capture states. If one corner is empty, it's a shared liberty of both stones, which might make the leakage less harmful than it first looks.

I'm checking whether a diagonal neighbor can count toward the propagation: if at least one corner is one of our own stones, the two stones are connected, so the diagonal only causes leakage when both corners are opponent stones. That would make a single unit unable to express the OR over the two corners.

I'm weighing a diagonal shortcut that would let propagation cover more distance per layer, but it would leak between separate groups and hide real atari and capture states, and fixing that would need a separate unit for each diagonal. Spare width in the early layers is too small to pay for that. So I'm turning to a different angle: checking for false atari on large groups using a liberty count or group-size proxy instead of exact propagation.

I'm checking whether false positives cluster in large groups, for example groups of 8 or more stones. If they do, a discount for big groups might help, though it would also weaken the rare but valuable true captures of big groups. I want to test this against the data first, by splitting the one-liberty predictions into true and false cases by group size.

I'm weighing whether to use local same-color stone density as a negative term in the capture unit, so that large groups get a smaller bonus, and I want to check this against the data by comparing true and false one-liberty predictions across group sizes.

