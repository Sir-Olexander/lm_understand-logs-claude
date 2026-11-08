Head L0H5 treats the two special tokens as a 2x2 pattern: each token attends strongly to itself and repels the other. That makes the head's function something that separates token 2 from token 22 on both the query and key sides.

I'm weighing whether the head treats token 2 and token 22 as an axis to separate on, so that a query of one type favors keys of the same type and avoids the other. In Python, that could mean a newline query attends to the previous line's start, while a space query attends to earlier spaces. I'm leaning toward the Python reading for now, though I still want to check how the same pattern would look in English.

Now I'm checking whether the same newline-versus-space split would hold up in English, where the newline case has far less training signal. The attention magnitudes for the two cases are close, so the mechanism looks identical there even if the data is thinner. I'm also noting that the comp-4 token behaves semi-specially, with some attention to 22 and a repulsion from itself.

I'm wondering whether romaji would also fit, since Hepburn-style spelling of katakana loanwords could bring in letters like l, and a vowel-consonant alternation might show up as a split in how the head attends to letters.

I'm considering whether the letter frequency pattern matters here. In romaji, the common letters are the vowels plus n, k, t, s, r, h, and m, while l, q, x, and v barely appear. English has its own rare letters like q, x, z, and j, so both languages would produce an uneven spread of letter strengths.

I'm wondering whether a vowel-versus-consonant split would dominate this head's reading of romaji more than it would for English or Python, since romaji alternates between the two almost deterministically. If that holds, the prev-token head should show a much stronger binary direction for vowels against consonants there. I'm also recalling a few groups of letters from earlier that separate positive from negative attention, which might reflect something similar.

I'm going to check whether the letter embeddings split into a dominant binary axis, which would point toward romaji. I'll take the Gram matrix of the 26 letter directions under a few different reading maps and see whether the top eigenvector isolates a group of roughly five or six vowels.

I'm checking whether the vowel-versus-consonant split shows up as a dominant axis under the induction head's query-key reading, since English would show a milder version of it. If romaji is the corpus, that binary direction should stand out much more sharply, so I'll compare how strongly it dominates across a few weight matrices.

