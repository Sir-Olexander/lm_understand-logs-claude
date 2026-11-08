Layer-1 heads that query and key both read the f-plane (o4) show very large per-dimension enrichment, between 1000 and 2500. That is far bigger than the 0.04 to 0.07 fraction I had earlier, which only looked at the share of the QK norm, since o4 is two-dimensional. Another group, where queries read L0H3's output and keys read the f-plane, comes in at 400 to 1000.

I'm now thinking that the layer-1 heads likely score each key by its absolute position, the query's own position, and the position of the last uB token, which would limit attention to the span from that last uB token up to the current position. I'm also wondering whether L0H3's output lines up with the same positional coordinates as the f-plane, so I'm checking whether o3 matches the slow positional component that L0H4 averages.

So if the layer-1 heads attend from the last uB token up to the current position, uB being a space or a newline would mean they gather the current word or current line. That would fit a shuffled-text setup, where all the query needs is a bag of the current line's tokens. The consistent alignment patterns in the later layers would be hard to explain under that reading, though, so I'm not sure it holds up yet.

I'm weighing whether uB is a space or a newline, since the L1 heads would gather the current word in the first case and the current line in the second. The strong q:o3 enrichment for those heads doesn't settle it yet, and the word-level reading fits English well, with each head pulling in the start of the current word.

I'm weighing whether L0H3 really tracks the most recent uB token closely enough to give word-level recency, since a space would be only a few characters back while a newline could be dozens. That precision matters: copying coarse positional info would blur i-3 against i-5 for words, but would be fine for lines.

I'm checking whether L0H3's output direction overlaps with the f-plane that L1 reads, and whether it draws on the same slow positional components as L0H4. If it does, the layer-1 heads would be comparing coarse positions, which suits newline-delimited structure better than word-level spacing. The slow plane carries large eigenvalues, so the precision question is still open.

I'm working out that if the M5 prev-token head uses an angle around 52-58° per position, plane A isn't slow after all, so the averaging in L0H4 would give a decaying signal that tracks early positions. L0H3 copying plane A from the last uB position would then carry fine relative phase, which suits word-level distances.

I'm checking whether plane A is actually fast: if M5's offset-1 head uses about 52° per position, then the averaging in L0H4 shrinks toward a 1/t signal. The large OV gain on that plane fits that reading better than a slow plane would.

I'm now asking what the layer-1 heads do with the large f-plane enrichment, given that f is a running average that shrinks with position. If that's right, the signal is strongest near the start of the sequence, so the f-plane probably acts as a sink or an early-position detector rather than a general-purpose positional code.

I'm going to check whether o3 and o4 share output directions, whether L0H3's inputs line up with plane A, and how the layer-1 heads read the query side.

