I'm looking at the top singular directions of the QK change, which seem to pick up digit tokens on the key side. On this head, L12h3 sv0, the key axis lights up on digits while the query side has digit-like features too, so the attention scores along that axis should drop, which means less attention toward digit keys.

I'm checking whether the digit-key attention drop in these heads is really about the operands or about the guessed number. Since guess and operands are both digits, the raw token projection can't separate them. The QK relative changes in layer 12 and layer 9 are only about 2–3%, so I want to rank all heads by their relative weight change to see which ones actually moved most.

I'm weighing whether the per-head attention shifts come from the layer's own weights or from changes inherited from earlier layers, and I don't have hidden states to tell them apart. What I can do is compare attention logits between guess and operand keys, which gives the total effect but not its split. So I'm leaning toward a weight-level ranking of QK change per head, combined with how digit-like each change is, and I'm also considering whether the guess is distinguished by its surrounding "is"/"it's" context.

I'm comparing where the operand attention shifts between the two models, and whether B's stronger pull toward digits at L9h0 and L12h0 accounts for the drop I see in A. Since I only have about nine runs left, I'll rank all 72 heads by their QK and OV change and check that ranking against the attention patterns.

I'm moving on to the MLP side, looking for individual units in layers 16 and 17 whose changes project onto digit tokens, to see whether they encode copying the guess or computing the sum. I also want a targeted attention check on the earlier heads at the punctuation after the guess, to test whether the guess gets marked there.

Next I'm weighing the L15 copy heads, which have large OV scores but barely attend to digits at the answer position in either model, so they likely don't matter here. The main routing candidates are L12h3, L10h2, L12h1, L12h0, L9h0, L9h1, and L7h0. After those, I want to look at MLP units in layers 16 and 17, comparing per-unit change norms across the gate and up projections.

I'm now going through the MLP units in those late layers: for each one, measuring how much its gate, up, and down weights changed relative to their size, to see whether the change is concentrated in a few units. I'm also checking whether the changed output directions line up with the digit unembeddings, and whether those units read digits in the first place.

I'm now setting up a full sweep across all 72 heads: the relative QK change per head, plus the OV change normalized the same way, so I can compare them against the attention evidence. I'll also check how much the copy-direction output shifts in each head.

I'm now going through the MLP units in each layer: taking the top eight by combined relative change, checking how much of the total change energy sits in the top one percent of units, and projecting their output changes onto the digit tokens.

