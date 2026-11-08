Layer 0 shows a clear pattern: for heads h0, h1, and h3 the change is negative, around −2 to −3.4e-3, against baselines near −0.004 to −0.016. That is roughly a 20 to 80 percent relative shift. So in this layer, those heads attend less from answer-type tokens to marker tokens than to other tokens.

I'm checking whether this layer-0 drop is specific to answer-type queries or applies to all queries, since a uniform change across distances could just mean the marker keys shifted generally. If it's generic, it would still suggest the model in A attends less to opinion phrases like "I think" or "guess," which would fit the correct-model reading.

I'm going to check whether this drop is specific to the answer-type tokens or shows up for generic query words too. I'll also compare the opinion words like "think" and "guess" against very common words like "is" and "it", and set those against random key-set contrasts to see how big the change really is.

