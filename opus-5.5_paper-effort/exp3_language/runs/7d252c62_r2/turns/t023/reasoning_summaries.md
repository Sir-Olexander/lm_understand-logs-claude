The correlation is trivially the identity, since ICA-whitened sources are already uncorrelated, so it tells me nothing. Moving on to the query-token effects for L0H1 and L0H5: a newline query gives the strongest signal at +14.7, a space query is negative, and most letters push negative while a few are mildly positive. So when the query is a newline, the positional preference shifts sharply, apparently in the same direction as tokens 7, 10,

I'm checking whether L0H5 at a newline query attends to the previous newline rather than the previous character, which would fit the sharp shift in the positional preference. The idea is that at a newline the next token is indentation, so the earlier line's start could carry useful information, though the value vectors may not carry much of that.

I'm weighing whether the strong newline effect means the prev-token heads spend their top TP direction on newline queries, which seems plausible given how large that signal is against the mostly negative letter values.

I'm wondering whether the letters 7, 10, 18 and 19 behave like newlines in the positional heads, since they share newline's sign on the top direction and in L0H5's position-token values. Newline is frequent but not uniquely so, so I can't yet tell whether that similarity matters.

I'm weighing the case for Python over English: the newline token has its own dedicated structure, with its positional-head key norms close to space's and the strongest modulation in the previous-token heads, and the first layer's heads look devoted to positional machinery. Against that, I still need to check how the English evidence stacks up.

I'm now considering whether space queries show the same kind of positional shift: L0H5 is strongly negative at a space query, and if the head is a previous-token head, at a space it might prefer earlier spaces, which in English would mean jumping back to the previous word boundary. In Python indentation runs that same preference would instead follow the spaces of the indent, so the space pattern might be a more direct test of which corpus the head is serving.

I'm comparing how much the positional heads devote to newline versus space: their token-modulation coefficients come out roughly equal, even though newlines are far rarer in English text than spaces. That suggests the newline structure is more than an English-corpus artifact, so I'd expect it to be weaker if the model were trained on prose.

I'm leaning toward Python at roughly 0.55, but the equal-and-opposite values on space and newline could just be a geometry artifact of the functional rather than evidence of a corpus-specific structure. A more direct test might be whether some head counts indentation spaces since the last newline.

I'm considering whether the positional heads might respond to spaces the same way they respond to newlines, which would be a clue about the corpus. I'm also wondering whether L1 heads favoring newline keys reflects paragraph starts in English prose, which would weaken the Python case.

I'm now asking whether an English corpus would ever produce a space right after another space, since collapsed whitespace would rule that out, whereas indentation in Python makes it routine. That gives a possible test: a head that reads "previous token is a space" at a space query would only be useful for Python.

I'm checking whether the previous-token head's value read of space carries "previous was space" strongly enough to matter for indentation, since that would be very informative in Python. But the value vector looks weak on space, so the signal might instead come from the landmark heads tracking distance to the last space.

I'm weighing the usual signatures for each corpus: shuffled text should lack prev-token and induction structure, romaji would show a smaller effective alphabet, Python would show strong newline and space structure, and English would show a full alphabet with rare newlines. Given how strong the newline structure is, I'm leaning toward Python, though I still want one more test on the indentation side, looking at how the landmark features resolve relative to newlines.

I'm checking whether the layer-1 heads treat newline keys differently depending on the query, since they seem to favor newlines at non-newline queries and avoid them at newline queries. I'm also looking at L2H2, where the newline and comparison terms stand out. Next I want to test whether the first layer carries the previous line's indentation forward from just after the newline, which would be a strong sign of Python.

I'm checking whether the newline-query induction pattern holds: at a newline query, a head could attend to the key just after an earlier newline, which would copy the first character of the previous line, and an indent space there would be informative. L0H5's output at those positions carries the newline identity, which would make that copy possible.

I'm settling on Python at about 0.55, English at 0.33, and romaji at 0.12, and I'm wondering whether romaji text could produce the same signals. Aozora-style texts indent each paragraph with an ideographic space, so that might explain why the newline-match induction is active without it needing to be Python.

I'm now thinking about whether romaji could produce the same newline pattern. Aozora-style texts start paragraphs with a full-width space, which normalization may turn into a leading space, and dialogue lines open with a bracket that also becomes a space. That would make newlines followed by spaces common in romaji too, so the newline structure alone doesn't rule it out.

I'm wondering whether the romanizer could map small kana or katakana loanwords to English spellings, since cutlet tends to do that, which would let letters like v, l, q and x appear in romaji text at a low rate.

Romaji can't be ruled out on alphabet size alone, since loanwords could bring in rare letters, but those letters would still be far rarer than in English. The stronger signal to check would be the vowel-heavy consonant-vowel rhythm of romaji, with 'a', 'i', 'u', 'o', 'e' making up roughly half the letters and 'n' appearing very often, plus the sh, ch, and y digraphs.

I'm testing whether the romaji hypothesis holds by looking for a vowel cluster in the prev-token head's token groupings. If one shows up, I'd expect the vowels a, i, u, e, o to stand apart from the consonants, since romaji alternates them so regularly.

I'm comparing the letter-direction strengths across the 26 letter components, and they all fall between about 0.16 and 0.31. If romaji were the corpus, rare letters like l, q, and x would be far weaker than the rest, so a flat spread like this leans against it.

I'm checking whether the 26 letter components all reflect letters that actually show up often enough in the corpus. If a letter were nearly absent, the model would have no reason to train a dedicated neuron for it, so the presence of 26 sparse letter components would argue that every letter, including rare ones like q and x, is common enough to matter, which would weigh against romaji.

I'm checking whether the rarest letter components have enough dedicated neurons to suggest those letters are common in the corpus. The high-kurtosis components with only 7 to 10 neurons each still got trained detectors, which would mean even the least frequent letters appear often enough. That fits English well, where q, x, z and j are rare but still plentiful across the training data. For romaji, 'q' would have to be far rarer, around a thousand occurrences, which seems too thin to earn a dedicated neuron.

I'm weighing whether newline-attention in these heads looks more like code structure or prose paragraph breaks, since both would give the newline a strong pull. A head that just attends to the most recent newline is a common pattern in code models, so that leans me a bit further toward Python over English.

I'm weighing comps 2 and 4 together, since both sit at odds with the layer-1 heads and show L0H0 key preference, so the space direction might be spread across them. Python still looks most likely at about 0.55, with English at 0.35 and romaji at 0.10.

I'm weighing whether the letter-frequency profile could separate English from Python, since Python identifiers lean on letters like f and l more than English prose does. Without a way to read off the letter identities from the embeddings, though, I can't compare against those frequency rankings yet.

Still working out whether the L1 heads look like induction heads: in English char models they usually show up in layer 1, but here layer 1 has only position-style heads, with the induction behavior appearing in layers 2 and 3.

I'm checking whether the L1 heads look like word-level attention: they avoid spaces, use the landmark features to stay within the current segment, and avoid repeating the same letter. That would fit English spelling, where aggregating the letters of the current word is most useful, and it would fit Python identifiers just as well.

I'm going to check how strongly newlines show up compared with spaces across several parts of the model: the MLP0 neuron loadings for comps 2 and 22, the read norms of the layer-1 Q, K and V weights on those tokens and on the letters, and the fc1 reading. If newlines are as prominent as spaces everywhere, that would point to a high newline frequency, which I'd compare against the expectation for English.

