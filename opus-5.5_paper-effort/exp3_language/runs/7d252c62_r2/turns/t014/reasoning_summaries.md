I'm checking the raw weight directions: dirs 0 and 1 look positional, mostly used by L0H4. Dirs 2 through 29 carry only small positional fractions and seem to be used by the copy heads, so they're likely token directions. That's 28 of them, and then there's a gap in the values before dir 30.

I'm working out how many token directions there are: 28 in the dirs 2 to 29, with the copy heads reading them. Since each token would need its own direction, 28 fits an alphabet like English or Python, while romaji would give closer to 24. The L0H1 spectrum also drops off around dim 28, which supports that count.

I'm counting the directions again: indices 2 through 29 make 28, with 0 and 1 positional. Working through the romaji alphabet gives about 24 tokens, which is short of 28, but the count of strong non-positional directions should track only the tokens that actually occur, so I'm checking whether any positional directions are being mistaken for token ones.

I'm checking whether the directions past the positional ones could be English or Python: with 28 token directions, the question is whether the alphabet is larger than the 26 letters plus space and newline, or whether Python's indentation and symbols would show up as extra tokens.

I'm now weighing Python against English: Python's indentation would mean long runs of leading spaces and frequent newlines, far denser than English prose. Repeated identifiers would also make induction heads matter more. Indentation could be tracked by attending back to the previous newline and counting the spaces before the first non-space token.

I'm now looking at the L0H0 and L0H3 heads, which seem to query on position and key on token identity, possibly locating the most recent occurrence of some particular character. Whether that character is a space or a newline would decide if they track word starts or line starts, and the newline direction is hard to pick out from the weights alone.

Now I'm checking how to tell the space token from the newline token, since their frequencies differ and the weights alone might not reveal which is which. I'm also considering whether the indentation runs in Python would show up as a distinct positional pattern in these heads.

I'm weighing whether the spectrum of the copy-head value vectors could separate English from Python, since letter frequencies differ between them, but the skew looks similar enough that it doesn't settle anything.

I'm looking at how L0H0 pulls from positional information and keys on token identity to find the last occurrence of some character, and I'm wondering whether its output sits in the same special dimensions that L0H4 uses as an attention sink.

I'm now working through the L0H0 attention in more detail, splitting its query-key product into position-to-position, position-to-token, and token-to-token blocks to see whether its token preference is real. I'll also compare the same breakdown for L0H2 and L0H3, and check the bias terms and mean positional contribution.

I'm weighing whether the L0H0 and L0H3 heads could key on the newline token by using a rank-one block of the query-key product, or whether a space-versus-newline split would show up as separate directions. Since the weights alone don't label the tokens, I'm thinking about counting how many key-token directions each head prefers, and whether the count points to spaces, newlines, or both.

I'm comparing the induction behavior across layers: L1 heads don't act as induction heads, while L2H1, L3H1, L3H5 and L3H0 do. I'm wondering whether that pattern and the strong line-structure demands of Python would separate it from English.

I'm now checking whether L0H0's key side reads a specific token direction, such as the newline, to find the last newline. I'd compare that with L0H4, which looks like a uniform positional average, and with L0H3, which may do something similar.

