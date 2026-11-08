I'm noticing about twenty-one roughly orthogonal cluster centers, which could act as token-detector directions, with a few extra related clusters on top. That might mean token-specific neurons for many tokens, with rarer ones getting fewer. I'm now wondering whether these clusters can stand in as token proxies, where each one is a set of neurons responding to the current token.

I'm weighing whether the cluster sizes (25, 19, 10, 8, 7 and so on) track how important or frequent each token is. If they do, the clusters could work as proxies for token identity, and I could probe bigram-like structure through the first-layer neurons that respond to the previous token. I'm also considering newline frequency as a way to judge whether those clusters line up with particular tokens.

Next I'm checking whether the first-layer neurons split by current token, and whether a "space follows space" pattern shows up in Python indentation that never occurs in English. I'm considering whether the largest cluster is the space token, and then looking at each neuron's preference for the current token alongside its preference for the previous token through the L0H5 OV mapping.

I'm checking whether neurons that respond to the current token also respond to the previous one, which would mark a repeated-token detector. The current-token and previous-token preferences can both be read as functionals on the same token coordinates, so I'm comparing their directions by cosine similarity to count how many neurons line up.

I'm weighing whether repeated-token detectors would show up in both languages, since doubled letters are common in English too, so that signal alone doesn't separate them. I'm leaning toward Python because the copy and induction heads in layers 2 and 3 look heavily used, which fits code's reliance on repeated identifiers, though I'm not sure yet.

I'm also considering whether the newline and space behavior differ enough between the two datasets to separate them, since Python text turns punctuation into frequent spaces, and English word-end prediction depends on local context in a similar way.

I'm working out how strongly L0H0 is selective for token X, by comparing its logit difference for X against non-X tokens, which should scale with the query and key components along the relevant directions. If that selectivity turns out large, L0H0 is a clear X-detector.

I'm looking at how L0H0's selectivity compares with the positional heads: its 0.45–0.6 is several times smaller than L0H1 and L0H5, so it reads as a broad head with only mild token weighting. Next, L0H3 has moderate token-specific key weights and a negative position-pair term, and it writes to both the token and positional subspaces. Then I'm checking L0H2, whose positive position-pair terms suggest it attends to recent keys with similar positions.

I'm going back to whether L0H2 keys on vowels or some other class, since its value reads only weakly from position. Then I'm wondering whether the largest MLP0 clusters, around 25 and 19 neurons, could correspond to the space token.

I'm weighing whether a Python-specific feature, like long indentation runs, would show up mechanistically in the first-layer neurons, and I'm considering how the copy heads might handle matching across lines.

I'm working out whether a head keyed on newlines, with its value reading the newline's position, could track indentation counts. The L0H0 and L0H3 patterns fit that description. The same structure keyed on spaces would instead give word-length tracking in English, so it doesn't separate the two datasets by itself.

I'm checking whether the positional planes have a period, since a 180° angle could mean a parity pattern rather than recency. If H5 tracks offset one and H1 tracks offset two, the shared top plane would show that, so I'm working out what the angle implies for which keys get favored.

I'm checking whether the two positional planes really have the periods I assumed, since if H1 and H5 were tracking offsets one and two, their angles should differ by a factor of two, yet both sit near 55°. That mismatch makes me suspect the plane is elliptical rather than circular, which would distort the measured angles for both heads in a similar way.

I'm now seeing that the distortion from an elliptical plane would shrink both measured angles toward zero, so the matching 55° readings could hide different true periods.

I'm checking whether H1 and H5 really share a top plane, since their top-two key and query alignments differ a little. If they do, the matching 52–57° angles would point to the same offset, while their other planes would separate them.

I'm testing which of H1 and H5 is the offset-one head, since a period-two plane would put offset one at 180° and offset two at 0°, which points the other way from my composition-based reading. The induction-head evidence says H5 carries the current-token signal, so checking that against the angles is the next step.

I'm checking the value composition for these heads: L3H5 reads the current token far more than the previous-token outputs of L0H1 or L0H5, and L2H1 splits about evenly, so the standard pattern holds where the value copies the token at the attended position.

I'm checking the offsets with the induction-alignment argument: if H5 is offset one and H1 is offset two, both the current-token and the cross-head shifts line up at +1, whereas swapping them would give mismatched shifts of +2 and +1. So H5 has to be the offset-one head.

I'm now treating H1 as the offset-two head, since the composition shifts only line up that way. The eigen-angles conflict with a simple parity reading, but I'm setting that aside and relying on the composition evidence instead. Next I'm wondering whether the most common token's frequency could be estimated from the MLP0 clusters.

Now I'm weighing how the newline-heavy Python text would show up in the first-layer heads, since its lines are short and indented, while English paragraphs make newlines rare within a 128-token window.

Now I'm wondering whether Python's frequent line breaks and indentation would show up as copy behavior in the first-layer heads, with the newline-plus-space pattern making the match easy. I'm also weighing whether long-range copying of identifiers in code might set Python apart from English's more local structure, and I'm tallying which of the L2 and L3 heads look strongly copy-like.

I'm checking whether the few extra induction-style heads could separate English from Python, since I'd expect roughly two or three in an English character model versus four to six in a Python one, though that count is weak evidence. Another angle is the space token: in English a space is always followed by a letter, while in Python it can be followed by more spaces for indentation, which would change its next-token distribution, but that isn't something I can observe directly here.

I'm wondering whether Python's doubled-space tokens, from indentation, could drive a repeated-space detector in the first-layer MLP that English text wouldn't produce as strongly, so I'm going to check the MLP0 neurons for that.

I'm going to compare each first-layer neuron's current-token preference with its previous-token preference and check whether they align. If one token keeps dominating the repeat-style neurons, that could hint at a repeated-space detector driven by Python indentation.

I'm checking whether the cosine between each neuron's current-token preference and its previous-token preference holds up, since the embeddings aren't orthogonal and a high match might only approximate the same token.

