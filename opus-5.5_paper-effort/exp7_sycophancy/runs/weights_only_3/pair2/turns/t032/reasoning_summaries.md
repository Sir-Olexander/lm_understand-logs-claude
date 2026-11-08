I'm noting that these attention profiles are mostly a constant offset across all distances, which shifts how much the head attends to tokens sharing the same μ versus the BOS sink, with some local bumps reflecting positional preferences. Now I'm looking at the base positional bumps near distances 5–7, starting with L7H1's peak at distances 6–7.

Now I'm going through the base-condition bumps for the remaining heads, most of them peaking around distances 6 to 8. Several of these heads show the same positional preference, which I'll weigh against their copy or anti-copy behavior.

Checking the L7H1 and L9H0 numbers, the local bump at distance 6 holds up in both the scaled and raw values, so attention there is pulled toward that offset for L7H1. For L9H0 the dip at distances 5–6 is much deeper than the drop at 10, which means it attends less to that band and copies the guess less.

Now I'm checking L8H2, which peaks at distance 6 in the base condition: its copy-related attention drops steadily from distance 0 to 10 with no local bump. L9H2's dip at distances 5–6 is also pronounced, which means it attends less to that band and copies the guess less.

Now I'm checking L8H0, which has no local bump at distance 6; its copy-related attention just keeps falling off from distance 0 to 10. L10H0 looks different: its attention rises at distances 6–7, so it leans toward the guess there and copies more.

Now I'm checking L10H1, where the attention dips slightly at distance 8 and otherwise falls steadily, so there's no real positional pull toward the guess. L12H3 shows a rise from distance 5 to 7 that stays elevated, which suggests a weak copy tendency. L13H2 is next, with its values near distances 0 and 5 still to be compared.

Now I'm checking L14H0's values, where the bump sits near distance 8, and then L15H1, where the local rise around distances 6 and 7 is only slight. Next up is L15H2, whose copy-related numbers I haven't compared yet.

Now I'm checking L15H2's bump at distances 6–7, which the numbers suggest pulls attention toward the guess, and then L17H1, where the base bump sits at distance 6.

Now I'm comparing the local bumps at distance 6 for the last few heads: L17H1 holds a clear peak there relative to its neighbors, while L17H2 is still rising through distances 6 to 10. L17H3 shows a slight dip around distances 6 and 7.

Now I'm checking L7H0's values, where the attention dips around distances 8 to 10, so it attends less to the guess and copies it less. L7H2 is next, with its values from distance 1 onward still being compared.

Now I'm checking L8H1, where the bump at distances 5 and 6 is only slight, so it copies the guess a little more there. Then L9H1, where the attention dips at distances 6 and 7 compared with distance 9, so it attends less to the guess there.

Now I'm checking L11H1, where the values dip at distances 5 and 6 compared with 3 and 9, so it attends less to that band and copies the guess less there.

Now I'm checking L10H2, where the values dip around distances 4 and 5 in both the scaled and raw numbers, so it attends less there and copies less. L10H3's attention falls off at distances 8 to 10, and L14H1 just keeps decreasing without any local bump.

Now I'm going through the remaining heads' values around distances 4 to 9, checking which ones show a dip or bump near the guess offset and whether that matches their copy or anti-copy role. L16H2 has a bump at distance 6, which weakly pulls attention toward the guess.

Now I'm tallying each head's local bump or dip at the guess offset and how its copy or anti role turns that into a push toward or away from the guess. The strong dips in L9H0 and L9H2 come out as the clearest suppressions, while L10H0, L15H2, L17H1 and L7H2 lean toward promoting it.

Now I'm sorting the remaining heads by whether their local bump or dip around distance 6 pushes attention toward or away from the guess. The mid-layer heads look like they reduce promotion, while the later heads seem to increase it, and I'm not sure yet what that means if the response spans multiple tokens.

Now I'm weighing whether the guess offset sits at Δ6 or closer to Δ9 once the answer is a multi-token phrase, and whether the mid-layer and late-layer heads push attention in opposite directions. I'm working toward a rough weighted sum of copy strength times local bump at Δ5–7 to see which way the net effect leans.

Now I'm going to compute each head's local bump at the guess offset, averaging the values near distances 5 to 7 against the values at distances 1, 2, 10 and 11. Then I'll weight those bumps by each head's copy strength to see which way the total effect leans. I'm also wondering whether the shared μ direction actually tells me anything about positional attention without the content.

Now I'm going to compute the weighted sum of the heads' local bumps at the guess offset, checking whether the net effect leans toward the guess or away from it, using a permutation test to see if the pattern holds.

