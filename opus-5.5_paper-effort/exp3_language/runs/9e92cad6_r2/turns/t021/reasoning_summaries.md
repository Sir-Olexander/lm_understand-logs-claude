I'm looking at the L0H3 eigenvalue angles, where most of the large-magnitude ones sit between 35 and 85 degrees, with a few smaller ones near 120 to 160 degrees. For a previous-token head at offset 1, I'm checking whether the angle should match the frequency component, assuming the positional embedding planes are pure sinusoids.

I'm weighing whether L0H3 is a sharp local head peaked at offset 1, since the dominant 35-60° angles would give a width of about three positions. Offset 2 would also fit the angles, so I can't tell them apart yet. The variance in learned embeddings tends to sit in low frequencies, which makes the offset ambiguous.

I'm weighing whether a "self" head could fit instead, with a peak in the future masked out so attention lands on the current token. That seems unlikely, since a symmetric positive query-key setup would do the job without such a rotational pattern. So I'll settle on L0H3 as a previous-token or local head.

I'm now weighing whether the model is mainly a local-context model, since L3H3 looks like a similarity or induction head and L3H4 is broad and uniform with a large OV. Romaji's local structure could explain the focus on nearby characters, though English or Python text would make copying from farther back more useful, so I'm not sure yet. I'm also considering a shuffled-text setup, where bag-like heads over recent tokens and line windows would matter more, and I want to look for evidence of line handling in the heads.

I'm going to check whether the key-only vectors across the heads share a token preference in the token subspace, such as a bias toward or away from newlines. I'll compare the projections onto that subspace between heads, and also against the value direction for the same group.

