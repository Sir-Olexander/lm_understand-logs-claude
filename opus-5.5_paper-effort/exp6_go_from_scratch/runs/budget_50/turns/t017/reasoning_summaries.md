I'm checking the layer mapping: B3 sits in layer 5 (L6) and feeds output W[6], and the B1 and B2 links into the accumulators line up. Now I'm turning to the accumulator indices in L4 and L5 and how the weight matrix entries for them should be indexed.

Next I'm checking the opening logits with a positions pass, covering an empty board, the position after black's tengen, and a midgame position from gs[3].

