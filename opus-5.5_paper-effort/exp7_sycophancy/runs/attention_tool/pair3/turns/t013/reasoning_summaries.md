The MLP changes look spread fairly evenly across units rather than clustering in a few places, since the top 20 out of 2048 only account for about 1.5 to 2 percent. A handful of units stand out as outliers, though: layer 7 units 256, 186 and 1334, layer 10 unit 404, layer 8 unit 706, and layer 4 unit 13. Layer 17 unit 595 ranks highest on both measures.

Now I'm comparing how the QK changes are distributed across heads, and L11h0 and L11h1 stand out: their attention to digits at the answer position is nearly zero, so their change doesn't show up there. I'm wondering whether they matter at other positions instead. My plan is to measure each head's average attention shift between the two runs over all query positions, to see which heads change functionally rather than just in weights.

