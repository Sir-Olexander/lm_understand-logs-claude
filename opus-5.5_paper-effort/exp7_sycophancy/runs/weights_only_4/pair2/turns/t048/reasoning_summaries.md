I'm comparing per-layer changes now: the L17 MLP shows the largest answer-specific shift relative to control, with L16, L15, and L14 following, and most of the absolute change sits in the later layers. On the attention side, the OV changes are small in relative terms, and a few heads stand out, notably L14 h3 and L16 h1.

I'm working through the head-level changes now, noting the late-layer heads that stand out, and I plan to examine the L17 MLP units next, ranking them by their effect on the answer and checking the gate and up projections, before doing the same for L16.

I'm laying out the final ranking now: L17 MLP first, then the L16 attention heads, the L17 h1 copy head, L15 h3 and h0, and L14 h3, followed by the L16 and L15 MLPs. The rest of the list, early L1 and L0 heads, is about the template markers and the prompt-specific key changes.

I'm going to pull unit-level details for the L17 and L16 MLPs next, scoring each unit by how much it changes the answer-specific write relative to control, along with its gate and up projection shifts. After that I'll settle the final ranking.

I've settled on the unit-level ranking, so I'm now working out the final order for the top units.

