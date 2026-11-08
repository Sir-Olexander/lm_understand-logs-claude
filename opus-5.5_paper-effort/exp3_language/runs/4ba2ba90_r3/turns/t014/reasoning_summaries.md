I'm looking at the b-out block eigenvalues, which come in a complex conjugate pair, so there's a rotation structure in a 2D plane. The summed broad OV values point to roughly three dimensions, and I'm wondering whether this rotation shows up in the L1/L2 QK circuits, which reminds me of something.

I'm picturing b-out as a 2D circle, with the QK rotation turning it into a relative offset. The query-bias times key-b-out term is large in L1, so attention there depends on the key's b-out value no matter the query. That could mean a query-independent preference for early positions, a sink-like behavior.

I'm weighing whether the L1 heads are local or broad, since they share nearly the same structure with a rotation in the b-out plane, and different offsets would show up as different rotation angles.

I'm checking whether the L1 and L2 heads have different rotation angles in the b-out block, since that would mean different relative offsets, like a family of local heads. If the angles are all the same, they might be broad heads instead. I also want to know whether b-out traces a helix over position, and I suspect that a plain prefix mean would converge rather than rotate, so the broad heads might be recency-weighted instead.

Next I'll compute the rotation angles of the b-out block for each L1 and L2 head.

