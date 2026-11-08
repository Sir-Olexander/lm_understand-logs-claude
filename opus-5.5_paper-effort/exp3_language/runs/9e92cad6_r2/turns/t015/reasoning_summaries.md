I'm checking how the positional signal from L0H3's QK gets used downstream: layer 0 reads it heavily through its queries and keys, and its value path picks it up strongly too. After that it fades in the later layers and the MLPs, which seem to ignore position almost entirely. Later attention heads barely read position directly either.

Next I'm checking how L0H3's value path and MLP0 read the token and that head's output, since those seem to carry most of the bigram-like signal. Later layers appear to get position almost only through the smoothed "outs" code, and their keys lean on it heavily while their queries barely read position at all. That would make their position dependence key-side only, which looks like recency or early-position preference rather than a previous-token pattern.

I'm now weighing whether later-layer heads could do previous-token attention at all. Since their queries barely read position, query-dependent selection looks unlikely, and a key-only recency bias would tend to pick the current token rather than the previous one. That makes me doubt there are any sharp relative-position heads in L1 to L3, with L0H3 and perhaps L0H0 as the only real candidates.

Later-layer queries barely read position, so their attention can't depend on the query's own position, which rules out a clean previous-token pattern. Only L0H3 and L0H0 look like real relative-position candidates, and the induction-style circuit I was expecting doesn't show up here.

I'm working out whether the later-layer heads could have any sharp relative-position behavior, and I'm questioning whether a char-level model this size would really need several local heads for short n-gram context, given that only L0H3 and maybe L0H0 seem to fill that role here.

I'm wondering whether L0H3 is really a local head at all, since it might instead attend mostly to itself, with its rotation reflecting a self-attention pattern rather than a previous-token one. In shuffled text, nearby characters carry little beyond the line's bag of characters, though line boundaries still matter, so a cheap local read could still be useful.

I'm checking whether the position signal in the "ins" plane could be a local, previous-token read, since a rotation angle of about 52° would correspond to roughly 37 positions back, which is too broad to be local. Other planes may carry higher-frequency position signal, so I'll look at those before drawing a conclusion.

Working out whether L0H3's rotation angle lines up with a local offset, and I'm questioning whether a 52° shift in that plane corresponds to a tight window or a broad span of many positions. If the positional arc turns slowly, that rotation would reach back dozens of tokens rather than just one, so it might not be a previous-token head at all. I want to check the other planes for higher-frequency position signal before deciding.

I'm wondering whether the eigen-angles, all around 45° across planes with different frequencies, fit a sharp single-offset pattern at all, since that would need the frequencies to line up. A broad attention pattern might explain the similar angles better.

I'm reconsidering whether the later layers could support the shuffled-text reading at all: their queries barely use position, their keys mostly carry a smoothed line-level code, and no induction-style key composition shows up. That fits the idea that position isn't doing relative-order work here, so I'm leaning toward the shuffled interpretation.

I'm checking whether "ins" really carries position rather than token identity, since L0H3's rotational QK would fit either reading. A query-side read of ins, with the value side reading ins too, would give a weighted average of positions, with the selection sharpness changing as the query moves.

I'm checking whether "ins" is positional or token-driven: if it carried newline or space identity, MLP0 would read it heavily, but MLP0 reads it only weakly while reading the token strongly. That makes "ins" look like a positional plane, the same as P.

I'm confirming that ins is a positional plane, since L0H3's value path ignores it and the MLPs read it only weakly. That makes the L0 group look like it's averaging positions weighted by content, which could serve as a line-structure signal. I'm now asking whether the later heads might still do recency-style attention over content in an ordinary, non-shuffled text setting.

I'm considering that a key-only recency feature plus content could let a head attend to the most recent space, which would be a plausible as-is pattern rather than a shuffled one. Then I'm noting that the lack of an induction-style key composition from L0H3's output still counts against a standard as-is reading.

Now I'm checking whether later-layer heads read the MLP0 output: if their keys pick up that composed signal, the induction-style pattern would appear in the key space. I'm also planning to measure how much of the later heads' Q/K/V sits in the top-k subspace of W_proj.0 and compare that against a baseline.

Now I'm weighing whether line structure matters much here: in English Wikipedia or romaji text, a 128-character window usually holds zero or one newline, so that mechanism would be secondary. In Python, though, there are several newlines per window and indentation at each line start, which makes line structure more central.

For a shuffled Python window, the bag of characters in the current line plus the previous line should carry most of the signal. The open question is whether the model tracks the average position of certain token types to estimate the current line's indentation level.

I'm checking whether the later heads' attention logits are small and broad in shuffled text, with only the line-boundary bias standing out, versus sharper content-driven in as-is text. I'm also planning to inspect L3H3's duplicate-token QK, since the token reads on both sides look strong.

I'm checking whether L0H3's same-token style head, which reads the token strongly on both sides with a large OV, is just counting how often the current character has appeared earlier in the context. In a shuffled line, that count would tell me how many of that character remain, but a uniform attention over the current line already captures that, so a dedicated same-character head doesn't seem clearly necessary.

I'm weighing whether the later heads could implement a "most recent occurrence" pattern through key-side features alone, which would make the lack of positional reads in their queries less decisive. An induction-style match would need the keys to carry previous-token content, and that signal doesn't show up clearly, so I'm leaning toward reading this as a non-induction setup.

I'm checking whether a positive-definite QK on a shared subspace could only match a context to earlier occurrences of itself, which means the head can detect repeated context but can't copy the token that follows it, since the value at that position has no information past it.

I'm now weighing L3H3 as a possible induction head: its query-side reads of the current token could match keys that carry the previous token, since L0H3's output at each key position may encode the token just before it. The positive QK trace of about 0.71 points that way, though K only reads L0H3's output weakly, so this stays tentative.

Checking whether L0H3's value output lines up with the input token subspace: the overlap looks small, so it seems to write into a different subspace from the one it reads, which fits a previous-token style head.

I'm going to measure the induction-style score for L3H3 directly, comparing the query-token and copied-previous-token subspaces against its overall QK norm. I'll also check the duplicate-token score and run the same comparison across the other later heads.

I'm going to measure how strongly L3H3 might act as an induction head, comparing its query-token subspace against the previous-token information copied in by L0H3, and check its duplicate-token score as well.

