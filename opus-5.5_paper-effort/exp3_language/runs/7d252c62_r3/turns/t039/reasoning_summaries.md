I'm checking the strong dimensions in the L0H1 and L0H5 outputs: about 23 for the first and about 21 for the second. I'm guessing the effective count for the romaji input is around 23, allowing for one dimension lost to layer-norm centering.

Next I'm weighing letter frequencies for English against romaji to see which language the sequence fits, though this alone doesn't settle it. I'm turning to a different test: how spaces and line breaks are distributed, since Python code has frequent newlines and indentation runs while romaji text rarely does.

I'm now considering whether the model tracks indentation by counting spaces since the last newline, which would need a head that attends back to the most recent newline and reads off its position. L0H3 looks like a candidate for that, since its output carries positional information.

I'm now weighing a head that copies positional information from the most recent space or newline, which would let the model count word length or indentation width. That mechanism is what I'll test next, starting with the prev-token head's key-token preference.

I'm checking whether the spaces and newlines in the Python text could be counted from the most recent newline, which would explain how indentation gets tracked. I'm also looking at whether an attention pattern can flag repeated tokens, since Python code repeats heavily, so I'd expect strong matching there.

I'm considering whether the model might detect indentation by checking whether everything since the last newline is spaces, which would need a head that attends to the most recent newline or non-space character. I doubt I can confirm that without token labels, so I'm looking for another angle, such as how the frequencies of spaces and newlines differ across the corpora.

I'm now thinking about how the geometry of the token embeddings might show the vowel/consonant split, since romaji alternates those so strictly that the previous-token signal would mostly track whether the last letter was a vowel. English leans on that distinction too, but less heavily.

I'm looking at whether the induction heads' same-token pattern can be read from the eigenvalues of their symmetric QK part, since a perfect induction head should score every token as matching itself. If the spectrum has roughly 20 to 23 positive eigenvalues, that would line up with the token dimension count I estimated earlier, which would tell me something about the vocabulary size.

I'm estimating that the 24 centered token vectors span about 24 dimensions, since removing the mean only drops the all-ones direction if it lies in their span. The strong heads show roughly 21 to 23 dimensions, and a weaker tail of values around 0.25 to 0.36 could be rarer tokens like q, x, z, or j that are only faintly copied.

I'm wondering whether the prev-token head's key-token preference might be a newline signal, since a head like that could avoid attending across line breaks or always lock onto them. I'm also considering a different possibility: that Python's vertical alignment across short lines calls for a head that attends to the same column in the previous line, with an offset equal to the line length, which varies from line to line.

I'm weighing whether the induction-style repetition in Python, English, and romaji would look different enough to tell the corpora apart. Romaji's repeated particles and verb endings might give it a repetition profile close to Python's, which weakens that route. I'm also wondering whether position-in-line effects in L0H4 or H0 could help, though I don't yet see how to measure them.

I'm comparing the key-token direction of L0H3 against those of L0H1 and L0H5, which overlap little with it, so they likely track different tokens, probably space and newline. Next I want to work out which of the two directions belongs to which token, and whether the prev-token heads are reading the token that came just before.

I'm checking whether the L0H3 query-side vector is positional or constant across positions. If it's a constant bias, the head would score keys by token identity alone, which would explain the offset-2 pattern I'm considering.

I'm working out the per-token key preferences for the L0 heads, where H1 is about 0.032, H3 about 0.024, and H5 about 0.018, so they're all small. I'm also checking whether MLP0's strongest output direction, singular value near 10.9, marks something like newline versus space, which L1 would use to choose between attending to the sink and to content.

I'm thinking about whether the MLP0 neurons could encode bigrams, since each one reads the current token along with the previous two, and romaji's consonant-vowel alternation would constrain which pairs show up. Without token labels I can't check that directly, and clustering the attention vectors hasn't turned up any clean groups either. I could instead use the negative same-token pattern in L1 or the eigenstructure of the induction heads' token-matching matrix, but that only gives directions, not individual tokens.

I'm trying to pin down individual tokens by finding basis vectors under which several same-token matching forms from different heads are all diagonal at once, which is like joint diagonalization in ICA.

I'm trying to get individual token directions by jointly diagonalizing the same-token matching forms from several heads, since a basis that makes all of them diagonal would pick out each token's embedding direction. That would let me count the tokens and look at how the vowels and consonants pair up, though I'm not sure the forms are clean enough for this to work.

I'm going to try a generalized eigenvalue problem between two of the induction-style forms, working in the roughly 22-dimensional subspace where one of them is positive definite, so the eigenvectors give a dual basis of candidate token embeddings. Then I'll test whether a third form is diagonal in that basis, since if it is, I've recovered individual tokens. I expect some fuzziness, because the heads only approximately match identical tokens.

