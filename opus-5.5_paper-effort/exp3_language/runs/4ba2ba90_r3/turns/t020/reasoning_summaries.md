Checking whether coordinate 222 is a massive-activation dimension: the output projection concentrates on it, along with a few others, and the layer-norm biases of layers 1 and 2 are large there too. The broad layer-0 heads seem to write a very large value into that coordinate, which would fit the attention-sink mechanism where keys with

Next I'm working out whether the sink behavior comes from position 0: a broad head at the first position attends only to itself, so its output is distinct, while later positions average over many tokens. That would give the first position a unique large value in that direction, which later heads could use as an attention sink.

Sink-dominated heads in layers 1 and 2 would probably look "broad" under an attention-distance measure, since they spend most of their weight on position 0. Their query bias also favours keys with a positive value at coordinate 222, which might be how the sink is selected. Whether these heads are actually labelled as broad depends on how the evaluation measures distance, so I'm still checking that.

I'm going through the head labels one layer at a time: layer 0 gets mostly broad heads plus a previous-token and a local one, and layers 1 and 2 look broad throughout. For layer 3, head 0 has a strong same-token and same-previous-token pattern, which suggests it's tracking earlier occurrences of the current bigram, so I'm weighing whether it's a copy head rather than a broad one.

I'm checking whether the first position is the sink: its broad-head output is distinct because it attends only to itself, which could explain why later heads treat it as a massive-activation target. I'm also weighing whether L3H0, which keys on matching current and previous tokens, is really a duplicate-token head rather than a copy head, since the residual at the earlier occurrence doesn't yet contain the actual next character.

I'm weighing whether L3H0 is really a same-token head: its strong match on identical current and earlier tokens, with almost no induction-style match on the previous token, suggests it attends to earlier occurrences of the current character rather than to the character that follows them. That would make it hard to call a copy head, and I'm considering labelling it "unclear" or possibly "broad".

Now I'm checking whether L3H1's positive "same previous token" score is real or just an artifact of its prevOut subspace overlapping with the token subspace, since the two aren't orthogonal.

I'm checking whether the first position and the broad layer-0 heads are what create the sink in coordinate 222, and whether the heads that pile onto that sink should count as broad.

I'm wondering whether the output dimensions follow token frequency, with the rarest letters like q, x, l, and v being too rare to leave the strong singular values I'd expect. If strength scales as a power of frequency, the weak dims at 7 to 8 seem too large for those ultra-rare tokens, so the English z, q, x, j set may fit better.

I'm testing the idea that the weak output dimensions track token frequency: if singular strength scales as a small power of frequency, the rarest English letters (z, q, x, j) would fit the weak dims about as well as the Python or romaji sets would.

I'm comparing the weak output dimensions against power-law predictions for each candidate token set. Working through the numbers, the observed strengths of 8.35, 7.91, 7.46, and 7.17 fit the rare English letters better than romaji, which would leave several dimensions far weaker.

I'm comparing the observed count of strong dimensions against what romaji's frequency profile would predict: roughly 21 strong dims with f and p near 9 to 10 and l, q, x, v much weaker. The observed pattern, with 22 strong dimensions, fits that prediction less well than the English rare-letter hypothesis does.

Checking whether the weak output dimensions could be rare romaji letters: the strengths 7.91, 7.46, 7.17 and 5.83 would mean l, q, x and v sit about as strongly as English's rare letters, which seems too high for tokens that appear so seldom in romaji text.

Weighing English against Python for the weak dimensions: English's rare letters z, q, x, and j fit the four weaker dimensions better, provided the fifth, lower dimension is a special token rather than part of that set. Python would only account for three rare tokens, so it's a slight edge to English. I'm also wondering whether the newline frequency, around 0.2% in paragraph-based text, could shift that picture.

Now I'm weighing English against Python for the weak output dimensions, and the newline frequency could shift that picture. English's rare letters fit the strong-dimension count of 22 cleanly, while Python would leave one fewer rare token to account for, so English still comes out ahead.

I'm now checking whether idx 24 is really a special dimension or a token dimension for something like a newline, since its copy-energy ratio is a bit lower than the token dims but its kurtosis is high.

Weighing the three candidate token sets against the singular-value pattern, English's rare letters z, q, x, j look like the best fit, with the one lower dimension possibly being a newline or special token, though Python and romaji aren't ruled out.

I'm now going to check whether L0H2 attends to the immediately preceding token, offset 1, rather than offset 2, since that determines how the induction head composes its match.

I'm reasoning about L0H3 next: its query-key score seems to depend on position, so it probably favours recent tokens and works as a local head. Its output also writes into the subspace where coordinate 222 is massive, so it likely feeds the sink signal too.

Now I'm turning to the L1 heads, checking whether their positional pattern looks local. Their rotation in the plane where L0H2 rotates for previous-token lookup is similar, which hints they may also track the immediately preceding token.

I'm comparing the L1 heads' rotation angles against L0H2's across the same planes, and the signs look inconsistent from plane to plane, so I'll treat the angle alone as weak evidence.

Now I'm looking at layer 1's positional pattern: its strongest rotation sits in the same plane as layer 0's head, but the other large plane points somewhere different, so I can't cleanly tell whether it tracks offset 1.

I'm leaning toward calling the layer 1 heads broad, but the induction head's reliance on their outputs makes me think at least some of them gather information from the immediately preceding tokens, so they may be local after all.

Now I'm checking whether the layer 1 heads attend locally: their outputs may carry previous-token information that the induction head relies on, though I'm not sure yet whether that points to a prev-token role or a broader gather.

I'm checking whether layer 1's heads act as previous-token heads: the query-side previous-token signal should match keys carrying the same token, which would push attention to the position just before. To test that, I'll compare the positivity of that block for the L1 heads.

I'm now computing the positivity of the previous-token block, T^T OVp^T A T, for layers 1 through 3, to see whether those heads match a query's previous token against keys carrying that same token.

