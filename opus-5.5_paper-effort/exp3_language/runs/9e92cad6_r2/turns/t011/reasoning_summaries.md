Key reads are heavily loaded into the 2D subspace of the L0 group, most strongly at L1 (50–88×), then L2 and L3. Queries and values stay away from that signal, with values especially avoiding it.

I'm looking at whether the 2D key-side signal could encode something like position or line membership, since that would explain how attention gets set up in the later layers. The catch is that a key-only feature can't match "same line" on its own, because that needs the query's line id, and the query-side reads here are small. So I'm weighing whether the signal is more likely a recency-like feature instead.

I'm checking whether the uniform-attention idea could explain the positional signal, since averaging value vectors over the prefix would produce a position-dependent output. Then I'm weighing a key-side feature like "is in the current line," which would need query information to work, against simpler options like distance since the last newline or a line index.

I'm testing whether the 2D input subspace shared by these L0 heads encodes position, since that would fit the common pattern of positional attention, with the newline and space features as the other main candidates.

I'm considering that a key-only positional feature could give a recency bias: if the logit grows with key position, the softmax favors recent tokens no matter where the query sits. That would explain why the keys lean so heavily on this 2D signal while queries barely touch it.

I'm noticing that a key-only feature could give a recency bias or a line-index signal: if the key feature counts newlines up to each position, a large multiplier would concentrate softmax weight on the current line, which key-side information alone could plausibly achieve.

I'm considering whether the first position's embedding could be special, so that uniform attention over the prefix mixes in a constant term that scales as 1/(j+1). That would give the keys a second coordinate to encode line count as a fraction.

I'm noticing that the per-token signal here decays within a line, so counting newlines through uniform averaging doesn't give a clean step. I'm now checking whether the jump at each newline, roughly 1/(j+1), is big enough for keys to tell the current line apart from the previous one.

I'm checking whether the L0 group might instead focus attention on newline tokens through a key-side content bias, with the value readout just pulling out newline information. I'm also asking whether the L1 through L3 heads would all depend on one shared 2D key feature, given that char-level models usually have more varied heads like previous-token and word-boundary ones.

I'm now asking whether L0H3 is a self- or nearby-attending head, since its output feeds the L1 values and queries heavily in the shuffled-text case, while in as-is text I'd expect previous-token information to matter more.

I'm checking whether the as-is English case would look different: I'd expect previous-token heads in L0 and induction-style heads in L1 and L2, but none of that seems present here. The induction route might go through MLP0 instead, though those outputs are high-dimensional and harder to read. I'm also looking at L3H3 and L3H4 next, starting with L3H3, which has a symmetric positive QK and doesn't read the L0 group.

I'm checking whether L3H3 might instead be a duplicate-token head: the query and key would read the same token subspace, so it would pick out earlier copies of the current character. That's plausible for shuffled text, but I'm not sure it accounts for the L0 signal in the keys.

I'm checking whether the L0H3 OV input directions, the ones that would carry token identity if it's a previous-token head, overlap with the 2D key-side signal, and I'll compare them against the L0H0 OV input directions too.

Checking the 2D input subspace against the token-identity directions read by L0H3 and L0H0 shows almost no overlap, but a strong overlap with L0H3's query-key directions. That points toward the signal being positional rather than token-based. If the group attends roughly uniformly, its output would just be an average of position embeddings over the prefix, which is a function of position only.

I'm checking whether the L0H1 key-side signal picks out newline tokens, with the query's position scaling how strongly it attends. If so, the output would be a weighted average of newline positions, giving later keys something like a running line index.

I'm reasoning that a key-only feature could work if it tracks the mean position of newlines seen so far: it stays constant within a line and steps up at each newline, so a large query multiplier would focus attention on the current line.

I'm reading the 2D line signal as a mechanism for attending within the current line: a key-only feature that stays constant inside a line and steps up at each newline. That would explain why most heads' keys draw on it so heavily, while the values avoid it.

I'm considering whether the newline-based line feature is especially useful for shorter lines, since the 128-token window in Wikipedia-style English usually sits inside one long line, while shuffled Python has several short lines per window and line identity matters much more.

I'm weighing whether the corpus is Python, romaji, or English by looking at what each implies for the window: Python has short, indented lines and a steady newline rhythm, while the other two tend toward long lines. Since the line structure seems so central, the question is whether the position-within-line signal would matter more for code than for prose.

I'm wondering whether the L0H3 head is a local one: its rotational positional pattern attends to nearby positions, and its output feeds heavily into L1. In shuffled text, the useful local signal might just be whether a newline appeared a few characters back, which marks the start of a line.

I'm wondering whether this could be as-is Python instead, since line structure matters there too. But in as-is code I'd expect previous-token and induction heads to show up in later layers, and I'm not seeing any K-composition from L0H3 into L1 through L3, so that looks unlikely.

I'm checking whether the L1 keys read the token identity subspace or mainly the line signal, since that would tell me if the heads are doing a bag-of-line aggregation. I'm also comparing the query-side dependence on the L0H3 output to see if the queries are modulating attention by recent-character context.

I'm checking whether the key-side direction of this L0 group lines up with the token-identity subspace (the one L0H3's OV reads) or with the positional directions in L0H3's QK input. The overlap test should show whether the newline-tracking feature is token-based or positional.

