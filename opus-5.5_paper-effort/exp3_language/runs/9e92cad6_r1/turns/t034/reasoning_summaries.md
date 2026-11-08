I'm finding that the segment heads' key direction mostly lies outside T, mainly in P or elsewhere, so L0H1/2/4 attend largely by position with some token preference. That makes them rank-1 positional heads, with scores built from query and key position directions.

I'm working out how the position codes from L0 feed into the L1/L2 heads, which read them with a 70° rotation to get relative-position attention with an offset. So those heads don't need raw positions, only the smooth low-dimensional codes that L0 already produced.

I'm settling on 'broad' for L0H1/2/4, since their rank-1 positional attention would spread weight smoothly over many positions rather than focusing on a few. Next I want to find a better way to estimate the token dimension, possibly by reading the induction head's query and key directions, because the counting approach I was trying isn't pinning it down yet.

I'm weighing the corpus choice now: with the T-dimension around 24-26, romaji and English both fit, while Python fits least. Given that only one induction head is involved, Python looks less likely. Romaji and English remain the live options, and I'm asking what else in the data separates them.

I'm weighing whether the romanization used a morphological analyzer that splits words with spaces, since that would make short tokens like "wa" and "no" very frequent. Romaji's consonant-vowel alternation is strong enough that the previous-token head would matter for telling syllables apart, though English would need that too. For the induction head, the repeated items I'd expect are particles and names in romaji, versus "the" and "of" in English.

I'm checking whether the line-start and spacing patterns in the corpus can tell romaji apart from English, and I'm looking at the logit-space singular values for the copy directions, with a set of 23 token dimensions showing up in the induction head's output.

Now I'm weighing whether the logit directions line up with the vocabulary: if the induction-copied dimensions and the leading shared direction together make about 24 token dimensions, that fits romaji's roughly 24 tokens better than English's 28, though the shift-invariance question leaves the count slightly fuzzy.

I'm considering whether English would make the induction head copy rare letters like j, z, x, and q, since copying matters most for rare tokens that n-grams can't predict well. That would push the logit-dimension count toward English's larger vocabulary, but the copy head's enrichment in dims 26-28 is ambiguous because other heads write to those directions too.

I'm checking the copy map's singular values next: 25 look significant before a sharp drop, which would point toward a vocabulary of roughly that size.

I'm checking how the copy map's rank changes when I restrict its outputs to dimensions 2-24 versus the full set, since I still can't tell which output directions are actually token logits. I'll try measuring the rank of the OV map directly from the T-restricted input, without restricting the output.

I'm going to test the copy map's rank directly: multiplying the OV matrix by the 26-dimensional FC0-based token subspace and checking how many singular values stay significant. Earlier, with a 28-dimensional subspace, I found 24 copied directions, so this should show whether the count holds up with the other basis.

