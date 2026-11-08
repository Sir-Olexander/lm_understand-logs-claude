I'm working out that L0H3 reads mostly from a subspace separate from the positional one, so it likely attends by position but carries token information through a different subspace. L0H0 seems similar, with its positional attention and token-carrying inputs overlapping with H3's. That makes me think P is the positional subspace and T the token subspace, with these two heads moving token identity based on position.

I'm now checking whether the keys of H1, H2, H4 and H5 read the token subspace T, since that would decide whether their attention varies by token and not just by position. If it does, H5's averaged positions would pick out specific tokens, such as newlines, and show where they fall.

I'm guessing these heads attend to particular tokens like newlines and carry their positions forward, so the output tracks where recent line breaks sit relative to the current position. That would give a "distance from line start" signal, which later layers could use to attend within the same line. I'm weighing whether that matters for Python, shuffled text, or prose, since the local positional head looks unhelpful once the text is shuffled.

Now I'm trying to pin down the token subspace T by looking at the layer-0 input orthogonal to P, starting with how L0H3's output rank and W_v for layer 0 are structured there.

I'm estimating the token subspace's dimension by looking at which directions orthogonal to the positional subspace the layer-0 weights actually use, and counting them with a knee in the singular spectrum.

I'm going to try to pin down the token count from the OV rank. If L0H3 is a previous-token head, it should carry nearly all token distinctions, so its rank would be roughly the number of tokens minus one. The 16-18 strong dimensions plus a slow tail fit that, though rare tokens like q, x, z, and j probably contribute little to the loss and could be compressed away. Romaji would have about 24 tokens, and a previous-token head there would need the preceding consonant to predict things like "sh" followed by "i", so the count is hard to read cleanly.

I'm checking whether the line-position machinery is well developed, which would only make sense if newlines are frequent and line structure matters. Python has that, but English Wikipedia lines are long paragraphs, so a window usually holds at most one newline. I'm weighing whether romaji Aozora text, with its shorter dialogue lines, would justify four of six layer-0 heads being devoted to newline-position features.

I'm wondering whether H1, H2, H4 and H5 might attend to spaces instead of newlines, since spaces mark word boundaries and come far more often. If so, their output would track the position of the last space, and subtracting that from the current position would give the length of the current word so far, which is useful for predicting spaces in ordinary text.

I'm checking whether the key directions of H1, H2, H4 and H5 read the token subspace or the positional one, since that determines whether their attention depends on which tokens are present.

