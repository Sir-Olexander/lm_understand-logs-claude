I've found that 26 FC0 directions are T-enriched when using projections, so the token-variation subspace looks 26-dimensional. Index 26 is borderline, though: with ks=[18,...] its T-enrichment is 1.11 and its singular value is 4.46, only slightly separated from index 25's 3.99. I don't want to overread that, so I'll keep 26 as the working count.

The romaji reading would need around 24 tokens, giving 23 dimensions, which is much further from 26 than the 27 expected from Python or English. A stray token, such as a newline, could account for the gap. Another possibility is that the romaji text has enough Latin-alphabet words or loanword conversions to bring in extra letters like l, q, v, and x, which would also give 27 dimensions.

I'm weighing whether the extra letters could come from kana-to-romaji converters that map small kana to prefixes like "x" or "l", which would bring the romaji count closer to 27. The strong T-enrichment in dims 24-25 argues against those tokens being very rare, though frequency and enrichment may not track each other cleanly.

I'm still weighing whether rare tokens would show up as weak directions at all, since the learned weight along each token's embedding should scale with how often that token appears. Adam's per-parameter normalization might change that picture for infrequent tokens, which would make their embeddings behave differently than the frequency alone predicts.

I'm weighing whether rare-letter directions would show up as a clean gap or a gradual tail, since their learned weights should shrink with frequency. With 4 rare letters in English, I'd expect their directions to be weaker than the common ones, so a sharp drop after 26 dimensions looks less consistent with that picture.

I'm now weighing whether the romanizer's choices for Hepburn spellings, like "m" before b and p or macron vowels, could add or merge letters and shift the count away from the clean 27 I'd expect.

I'm weighing whether the Hepburn spellings would really add enough distinct letters to reach 27 tokens, and it looks like they don't. Counting the letters I'd expect from romaji lands at roughly 24 with whitespace, so the romaji reading looks less likely than English or Python.

Since there are no biases in the weight matrices, the constant mean direction μ can work as a bias, so the model may learn to read it even though it carries no token information.

I'm weighing whether English or Python fits the 26 robust directions better: both would nominally give 28 tokens, but English's rare letters might only show up as a weak tail rather than clean dimensions.

I'm now weighing whether English or Python better matches the 26 clean directions, since both would nominally give 28 tokens. Python's frequent tokens should produce a full set of robust dimensions, so if there are only 26, a couple of tokens may be getting dropped, though I can't yet say which.

I'm working through which letters a Python source would contribute at roughly what frequency: 'x', 'q', 'z', and 'j' each show up in a few common identifiers, around a few tenths of a percent, while 'k', 'v', and 'w' come in around a percent or less.

Looking at letter frequencies now, Python's rarest letters q, z, and j sit around 0.2 to 0.3 percent, while English's z and q drop below 0.1 percent. If the 26 robust directions are real, rare tokens at those rates should still form clean dimensions in Python, but in English the sparsest letters could fade into a weak tail. That makes English the better fit for a count of 26, though I'm not certain of it.

I'm checking whether two rare letters could be weak enough to fall out of the count, which would fit English's 26 robust dimensions out of 28 tokens. There are no faint extra token directions visible past dimension 26, though, so the picture stays ambiguous between English and Python.

I'm checking whether the singular-value spread within the 26-dimensional subspace points to any weak directions, looking at the value-side readers next.

I'm wondering whether the indentation runs in Python could show up in how the early heads handle repeated spaces, since a previous-token head would then be attending to whitespace rather than to a different character.

I'm considering whether an induction head at layer 3 is the only one, and whether the model skipped forming induction heads at layer 1 because the L1 heads are all local.

Next I'm checking which token the L0 heads lock onto, since newline-focused attention would point toward Python and space-focused attention toward English. I'm trying to find the space direction within the token subspace, maybe by looking at which direction is most frequent.

I'm now checking whether the segment heads L0H1, L0H2 and L0H4 key on one shared boundary token or on different tokens, by projecting each head's key direction onto the 26-dimensional token subspace and comparing them.

I'm wondering whether the segment heads (L0H1, L0H2, L0H4) are tracking indentation, which would point to Python, or word boundaries, which would point to English. Python's long runs of line-start spaces would demand precise position-in-line counting, and the three early heads plus the uniform head seem like a lot of machinery for that, though English word positions could need something similar.

I'm weighing whether the romaji reading holds up: if spaces only come from punctuation, word segmentation is less explicit there. The 26-dimension count is my strongest discriminating evidence, and it argues against romaji, which predicts about 23-24. Between English (expecting 24-27 given weak rare letters) and Python (expecting 27-28), 26 leans slightly toward English, though Python's rare letters could also be weak. Next I want an independent check from the logit-side token dimensions.

I'm looking for a way to tell English from Python apart, and I'm weighing a few options: how often newlines appear and how predictable line endings are, whether long runs of indentation spaces at line starts stand out, and how much repeated "self" would drive induction.

I'm weighing whether the distinguishing power is low here, and whether the vocabulary argument holds up for Python: after normalization digits fold into spaces, so the token set is essentially letters only.

I'm checking whether the count is 27 live tokens instead, since English with one weak letter or Python with none dropped would both give 27 dimensions. The dimension just past 26 has a singular value of 4.46 and a T-enrichment near 1.1, which looks like noise rather than a real token direction.

I'm checking whether the induction copy map from the 26-dimensional token subspace into the logit side has a clear rank, and I'm looking at its singular values to see where the drop-off falls.

I'm checking whether the two T dimensions the induction head doesn't copy are the space and newline tokens, since it may simply not need to copy those. That would leave the remaining 24 copied dimensions as the letters. I still can't tell English from Python from the copy count alone, so I'll look for a structural signal instead.

I'm asking whether the segment heads track line starts, which would suggest Python, or word starts, which would suggest English. For Python, a head that finds the previous line's start would need a clean position signal, and for English, the same mechanism could simply pick out the previous word.

I'm weighing whether to check the strength of the L3H3 induction head against the other heads, since its output norm would show how much the model leans on copying. Python's repeated identifiers would make that copying matter more than it does in English.

I'm comparing L3H3's output strength against the MLP3 contribution, where it looks moderate next to the larger W_proj.3 singular values. I recall induction-head formation being strong in code models, so the lack of early induction here leans me slightly toward English over Python, though that's a weak argument.

Next I'm looking at whether the layer-0 heads handle the newline token differently from the space token. Python's indentation runs would make the spacing a bigger factor than it is in English.

I'm checking whether layer-1 attention and the FC1 MLP also show a 26-dimensional token subspace, using their raw token reads as an independent test of that count.

I'm estimating the token count from the T-subspace spectrum and the reader projections, expecting roughly 26 ± 1. Counting alone won't separate English from Python, since both nominally give 28, and romaji's 24 looks unlikely unless Latin letters are present. I'm also considering whether the reading of the Japanese literature corpus changes that picture.

Cutlet, by default, turns katakana loanwords into their English spellings, so letters like l, v, x and q could appear in romaji text, though only rarely. That would mean a romaji corpus could in principle cover all 26 letters, which weakens the case that the 24-letter count rules romaji out.

I'm wondering whether cutlet's default Hepburn output would put macrons into the text, since normalization would turn them into spaces and split words oddly. I don't think the puzzle designer would have left that in.

I'm now wondering whether a romaji corpus could still contain all 26 letters, since loanwords like "curry" or "television" would bring in rare letters such as l, v, and x. If so, the 26-dimension count doesn't rule romaji out on its own. I'll need a discriminator that doesn't depend on letter inventory, like the vowel-consonant alternation that romaji's syllable structure would impose.

I'm weighing whether vowel-consonant alternation would show up in the dominant binary features, since romaji's syllable structure would make it stronger than in English, where a space marker tends to dominate. MLP0's spectrum doesn't have one clear top direction, with values around 12.8, 10.45, 10.1, and 9.06, so that check won't settle it on its own.

I'm now weighing whether the space and newline frequencies could separate Python from romaji, since Python's indentation runs would push newlines and leading spaces far higher than in English prose.

I'm weighing whether the newline and space handling in the layer-0 segment heads would tell me something, since frequent spaces in English could dominate attention in a way rare newlines would not. I'm checking whether the heads that look like line-position trackers are keyed on newlines or on the far more common space token.

I'm now going to test whether the space and newline tokens are treated differently by the layer-0 segment heads, since frequent spaces in English could dominate their attention in a way rare newlines wouldn't. Next I'm trying to find the direction that the segment-head keys read most strongly, and checking whether it lines up with the top MLP0 read direction.

I'm wondering whether same-token avoidance in the L1 heads is just how they skip self-attention, or whether it does real work on runs of repeated tokens like Python's indentation spaces. If it does real work, that would point toward code, since indentation runs are common there and rare in English prose.

I'm now asking whether the L1 heads' same-token avoidance does real work: they seem to read only the current segment, so within a word they would attend to the other letters, and within a line to the other characters of that line. The rotated outlier code might be how they tell where the segment starts.

I'm going to check how the outlier code dimensions are read by queries versus keys in L1, building the small QK block over those five coordinates and the query bias term that acts as a key preference.

