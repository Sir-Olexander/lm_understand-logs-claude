Summing the z-scores across all 72 heads gives +30.1, roughly 3.5 standard deviations above null, so the model copies numbers more at the token level after the change. The color and capitalization controls, and the column-capped version, don't show that increase, so it looks specific to numbers. Still, it doesn't track how strongly each head copied in the base model.

I'm checking whether the first-order difference between sycophantic and correct runs shows up as token identity for numbers in the heads attending to the guess. The value-input approximation is shaky at late layers, since the residual there isn't the raw embedding, but the digit directions in L15/L16 v_proj still look meaningful.

I'm going to use a couple of the remaining runs to pin down which components drive the number-copying increase, starting with per-head z-scores across all 72 heads to find the top contributors. Then I'll rank the candidates: the attention heads in L15, L14, and L11 that show the copying increase, the digit-focused query and key changes in L10 through L12 and L17, and the MLP units in L14 through L17.

I'm going to score each head on three things: its number-copying change, how much its query changed at digit inputs, and how much its output capacity for digits shifted. I'll combine those into a single ranking. The same comparison will go over the MLP units in L16 and L17, and maybe L15 and L14, to find the ones where digit-writing capacity changed most.

