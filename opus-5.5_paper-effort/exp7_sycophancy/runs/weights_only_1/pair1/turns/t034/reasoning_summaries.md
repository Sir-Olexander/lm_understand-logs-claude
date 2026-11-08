I'm checking the independent τ result: the L16/17 head copy sums to 24.2 against a null of about −0.4 ± 4.8, roughly 5σ, with all eight heads positive. L12–15 look mixed, though L15 heads 0 and 3 are clearly positive. I also need to remember that this τ was built from the L13–15 o_proj writers, so L13 may not be fully independent here.

Now I'm going after the mechanism behind L16 and L17, checking whether those heads attend to the guess and how their query and key changes line up. I also want to pin down what the L12 heads 0, 1, and 3 do in terms of their OV behavior, since their copy scores point in different directions.

Now I'm ranking the components by how strongly they carry the effect: the L16 heads come first, with head 2 the largest in absolute terms, followed by L17 heads 0 and 1 as global copy heads, then the L17 MLP as the main writer into the logits. I'm also weighing the L12 query change and the L16 MLP as the next candidates.

Next I'm checking the L17 down_proj's top singular direction: which units shift, whether that output lines up with τ, and how the gate and up projections change. I also want to see if L16's query or key changes point at the τ keys, and what the L12 rank-1 query heads 0, 1, and 3 write out through OV, possibly feeding later copy heads.

I'm checking whether L12 heads write something that the L16/L17 keys read, which would mark the guess, so I'll run a composition test between the L12 head outputs and the L16/L17 key weights, including the change.

