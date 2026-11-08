I'm checking which dimensions the shared V1/V5 directions use: 27 of them have strong overlap, and MLP0 reads those 27 strongly, while the rest look like random directions.

The 27 dims are read strongly by MLP0 at about the same level, including the weaker-shared ones, which suggests they're mostly token directions. That points toward English or Python (28 tokens) rather than romaji. But subtracting the mean should remove one dimension, which would leave 28 rather than 27, so I need to check whether the all-ones direction is in the token span.

I'm working out whether the 28-dimensional span forces a discriminative dimension count of 27: if the vectors are independent, one direction gives every token the same output and carries no identity, so 27 is what I'd expect for 28 tokens. That matches English or Python and doesn't fit the roughly 23 I'd expect for romaji's 24 tokens. I'm settling on romaji as unlikely, and now weighing English against Python.

I'm weighing English against Python now, with newlines as a possible discriminator, since English prose has section breaks and list items that Python code handles differently. The uA direction looks like the constant one: weakly read by MLP0 and not useful for identity, which would account for the 27 discriminative dims out of 28 tokens.

So the all-ones-like direction gets weakly read and copied, so it doesn't help identify the corpus. That leaves English versus Python, and I'll need to look for a different discriminator. Python's frequent newlines and indentation-copying behavior are one candidate: the model has to track the previous line's leading whitespace from the current position.

I'm now checking whether a head's key strongly favors newline tokens regardless of the query, since in Python that would help track line context. I'm also weighing whether uB, which the previous-token head copies and MLP0 reads strongly, is the newline-versus-space direction.

I'm considering whether a space-versus-newline split in the uB direction tells the two corpora apart, and it doesn't, since both are plausibly important in either case. The frequent-space pattern might point to English word starts instead, with a head sharply attending to the most recent space. I'm now wondering which corpus would push L0 toward two uniform positional heads whose outputs feed relative-position rotation in L1.

I'm checking whether the rotation angle in L1 implies a positional offset, since a 73° turn against a per-position phase of about 27° would point to attention reaching back two or three tokens. The plane-A angles might instead mean a slow frequency component, which would make that offset much larger, so I can't settle it from these numbers alone.

I'm checking whether the plane-A angles fit a short offset. If that plane has a slow frequency, both angles near 52–58° would point to attention reaching far back, which clashes with a previous-token head. Maybe plane A carries an absolute-position bias instead, and the offset story only holds in the other planes.

I'm testing whether a kernel weighted at offset 2 with a negative self term could explain plane A's angle, but that gives about 120° and doesn't match, so I'm setting the positional details aside and turning to whether the model is English or Python.

I'm weighing whether a head that attends to the same column in the previous line would show that the corpus is Python, since indentation alignment matters there and not in English. I'm also considering whether the number of live MLP0 neurons hints at a smaller n-gram vocabulary, though I can't calibrate that yet.

Now I'm weighing English against Python, and I'm wondering whether a head that spreads attention evenly over a run of recent spaces would be the signature of indentation counting, which would favor Python. Still, a local head that gathers the characters of the current word would fit English just as well, so I'm not settling it yet.

I'm probing whether double-letter bigrams, like "ll" or "ss", would separate English from Python, since Python adds runs of repeated spaces from indentation.

I'm checking whether the same-token neurons share a space direction, since that would point toward Python's long runs of indentation spaces. The count of 24 against a baseline of about 5 suggests they exist, but I haven't yet confirmed they cluster along one direction.

I'm now checking whether the neurons that read a token twice, meaning those with a and b pointing the same way, cluster along one shared direction. If they do, that would be a space-run detector, which points toward Python, since English double letters would be spread across many different letters.

