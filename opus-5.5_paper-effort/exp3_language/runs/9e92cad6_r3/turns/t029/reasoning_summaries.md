The singular values for W_proj.3 drop off sharply: one large component near 810 falling to 122, then a cluster of three around 70-80, then a few in the 20s and 30s, and the rest sitting at a bulk level of roughly 6-10. The two components at 14.3 and 13.0 sit just above that bulk, so the spectrum looks like one dominant direction, a small group, and a couple of borderline ones.

Now I'm working through the frequency classes for English text: the common letters and space form a strong group, and the rarer letters like x, j, q and z should sit at the weak end, which would explain the lower singular values.

I'm checking how the English letter frequencies line up with the spectrum: about 21 strong components, a few moderate ones, and four weak ones, which fits well. Now I'm running through approximate Romaji frequencies from Japanese text, with the vowels a, i, o, u and common consonants forming a strong group and the rare letters like z, j, f and p sitting lower.

Now I'm comparing Python's token frequencies, with spaces dominating from indentation and the common letters forming the strong group, and checking whether that fits the spectrum as well.

I'm weighing whether the Python token frequencies, with spaces and letters like f, l, k, v, w, q, z and j all reasonably common, could account for the same spectrum, though it would need roughly 24-25 strong components plus a few weaker ones. Still, these singular values come from mixed directions, so counting dimensions only loosely points to a vocabulary of about 28 tokens, which fits English or Python.

I'm checking whether the four extra dimensions near 21-40 could come from Latin letters embedded in the Japanese text. Aozora-style texts are mostly Japanese, but occasional English phrases would still give l, q, v, and x enough occurrences across a billion characters to get their own logit directions.

Checking whether those rare Latin letters in the Japanese corpus would actually earn their own directions: each one shows up only about 0.01% of the time, so the payoff from modeling them is tiny, though not zero.

I'm checking the weakest singular values, roughly 40 down to 21 against a typical level near 300, and comparing them with the moderate and rare letter groups in English. The newline could be a moderate contributor in English but is too frequent in Python to fit the weak end, so I'm setting Python aside here.

I'm testing a Python hypothesis now: newlines should be frequent because of short lines, and indentation spaces should dominate, so maybe 24 strong directions plus a few weak ones. That could fit the spectrum too, and I want to compare it against the English story by looking at induction-head behavior.

Looking at whether the spectrum fits English or Japanese romaji better, I'm weighing how strongly the rare letters register. The top output direction, carrying about 11% of the energy, might separate space from letters, which would be a word-boundary decision in English, a space-or-newline versus letter split in Python, or a vowel-versus-consonant split in romaji. I'm also considering whether the copy head's smaller weight on that first direction says anything, though that doesn't settle it yet.

I'm considering whether these L1 heads act as previous-token heads by attending to the most recent position that isn't the current character. In English, doubled letters are rare, so that skip-the-same-character rule would usually land on the preceding token. I'm wondering why the model would build that behavior here rather than rely on the learned positional mechanism that L0H3 already provides.

I'm trying to pin down whether the L1 heads really favor recent positions. If their attention scores were driven mainly by the start-of-sequence sink, they'd lean toward early tokens, which would argue against a recency behavior, unless the sign runs the other way and later positions are favored.

I'm checking the quadratic-form numbers for that 2×2 matrix: the trace and determinant give real eigenvalues near 0.78 and 0.48, so the map is real but strongly non-normal. Then I'm wondering whether the input embedding carries a constant offset as well as the rotation terms.

I'm weighing the head labels now: L0H3 as previous-token, L0H0 as local, and the L0H1/2/4/5 heads as broad averagers. For the L1 and L2 heads I'm still unsure whether they're local or broad, so I want to check how their key bias projects onto the output directions before deciding.

I'm estimating how the output direction Fout varies with position: it's roughly the H5 output averaged over earlier positions, and since the token embedding barely overlaps the input direction, the positional part dominates.

I'm considering how a positional signal in the Fin plane would shape the running average: a fast rotation would make the average decay quickly and favor the earliest positions, while a slow one would produce a smooth ramp. Then I'm checking what a 44° rotation of that plane would imply for where L0H3 attends.

I'm settling on L0H3 as the previous-token head, since a single offset across all its positional planes would be hard to get from mixed frequencies, and small models usually need high-frequency positional components for that job.

I'm weighing whether shuffled text would still produce a broad head that boosts letters already present in the current line, which would act like a copy operation. That would mean the copy-like behavior doesn't by itself show the text is in order. Next I want to see whether L3H3's content-based matching sets it apart from that kind of broad head.

Checking whether L0H3 is a sharp previous-token head or a broader local window, since that decides whether the order question even makes sense. If it's a local bag, then keys matching my token's neighborhood could help predict characters in similar lines, which would look like copying without the text being in order.

I'm weighing whether the L1 heads' "attend away from the same token" pattern makes sense under a shuffled-text reading, where the current character is known but its order is not. I'm checking whether that anti-match behavior, combined with the query reading its previous token, would reproduce the previous-token signal I already attributed to L0H3.

I'm checking whether L0H3's positional scores are sharp enough to count as a true previous-token head, estimating the typical size of the position embedding after layer norm and how much of it lands in that head's subspace.

Now I'm checking how L1's heads combine the query's previous character with the key's current character, where the strongest block is the one linking those two.

I'm checking whether the L1 heads' query-key block pairs my previous character with the key's current token. If keys matching my previous token are where I look, the value could read that key's own previous-token signal, which would reach back two positions. Next I want to verify the sign of that block.

I'm weighing whether L0H3's positional pattern is sharp enough to count as a true previous-token head, and I'm leaning toward as-is at roughly 0.75 confidence.

I'm checking whether L0H3 is a sharp positional head or a broader local window, since a shuffled-line model would more likely show newline-keyed attention and bounded line-level bags, and L0H3 instead looks like a purely positional offset head that copies token identity.

I'm leaning toward the as-is reading at about 0.8, since L0H0 also looks like a local positional head rather than a broad bag. For the corpus, the rare-letter dimensions fit English best, with Python plausible and romaji less so, and the single late induction head points away from Python's heavier repetition.

I'm weighing whether the top output direction, with about 11% of the MLP3 energy, points to a space-versus-letter split, which would fit English best, or a space-versus-newline split for Python. I'm also checking whether the next cluster of singular values hints at a five-way vowel subspace for romaji, though those values look smooth and don't clearly show that.

I'm now looking at L3H4, which has very small query-key scores but a large output that spreads almost uniformly across the context and pushes several logit directions. I'm wondering whether it works as a broad topic or unigram-style signal, and whether that kind of signal would point toward English, Python, or romaji.

I'm checking whether the rare-letter directions 24 through 27 are where L3H4's output lands most, since a broad head that boosts rare letters already in the context would fit an English corpus with foreign names.

I'm checking whether L3H4's value vectors pick up those rare letters, since its top output singular values are fairly large (21.3 down to about 6).

I'm checking the output projections of L3H4 onto the W_proj.3 directions, looking for how many dimensions carry most of its weight. Its query-key scores are small, so its attention looks broad, and I'm noting that its keys don't pick up the Fout signal.

Now I'm checking whether L3H4's value vectors read the rare-letter directions, and whether its near-uniform attention is the kind of broad head that boosts letters already in the context. I'm also testing whether the weak logit dimensions 24-27 could be rare letters like x, j, q, z, which would point toward an English corpus.

I'm checking whether the anti-same-token eigenvalues show one dominant direction, and I'm wondering if a dominant "space" direction would fit Python's indentation runs. Looking at the L1 heads, I see only L1H5 has a single clearly dominant eigenvalue, so I'm weighing whether newline frequency, around 3% of Python tokens versus far fewer in English, could tell the corpora apart.

Now I'm weighing whether the top output direction points to a space-versus-letter split, which would fit English, or a space-versus-newline split, which would fit Python. I'm also checking whether the L3H4 head's rare-letter directions could be a broad detector that suits the corpus, and I'm leaning toward English as the most likely, then Python, then romaji.

I'm comparing the output-direction counts against the vocabulary size: about 28 dimensions are in use, matching 28 tokens, but if romaji drops four letters, the last MLP wouldn't need to write to those dimensions, so I'd expect around 24 strong ones. Instead I see 21 strong plus a few weaker but still clearly above-bulk values, which is hard to read without token labels.

Now I'm weighing the 4 weak dimensions, which might be rare letters like j, x, q, and z, against the 24 strong ones. I'm leaning toward English, since the energy pattern of 24 strong and 4 weak fits that corpus well, while Python's flatter letter frequencies and frequent newlines would predict a different split.

I'm comparing the 4 weak dimensions against the strong ones: their energy is about 0.07 to 0.13 of the median strong value. For English, rare letters like j, q, x, and z would give a frequency ratio around 0.03 to 0.05 against typical letters, while romaji's foreign-only letters would sit much lower, near 0.002.

Now I'm weighing English against Python by looking for newline- or indentation-specific signals in the last layers, since Python's frequent newlines and long space runs would show up differently in the output directions.

I'm weighing whether induction favors Python: a 4-layer code model would likely build early induction heads, but only a late one shows up here, which fits English better.

I'm checking whether induction shows up in the L2 heads by routing prev-token information through the L1 outputs, though that path looks indirect. A more direct test is the positivity of the query-key term between the token and the L0 head-3 output, which gives a modest signal for L2H4 and L2H0 and a strong one for L3H3.

I'm settling on English as the most likely corpus, since a late induction head fits it better than the early induction behavior a code model would need. I'm now turning to whether Python's indentation could be the deciding signal, checking how the model predicts leading spaces after a newline by copying the previous line's indentation.

I'm checking whether the first head's attention depends on token identity, since its score scales with how strongly each key token aligns with a fixed direction. A newline token with a large projection onto that direction would get several times the weight, so the head would average over newline positions more than others.

I'm weighing whether Python's frequent newlines and indentation would show up in the first head's output, and I'm planning to test whether any L0 head attends to a particular token like newline, since the query-independent key terms are small for those heads.

I'm checking whether the first layer's head favors keys where the running average of the input features is negative, since that would make certain positions stand out.

I'm checking whether the first layer's large output mostly comes from one head, and whether the MLP's contribution is the same size. My next step is to compare that head's output norm against the other heads and the MLP output.

I'm checking how the first layer's output would look if the big component only appears at a few special positions, like position zero, where the head attends only to itself. That would fit the usual pattern of large activations on one or two tokens rather than on every position.

I'm reasoning that the first layer's large output likely comes from position 0, where that head attends only to itself and its output is strong. Elsewhere the averaged contributions should largely cancel, so this looks like a massive-activation marker at the start of the sequence, and later layers can use it as an attention sink.

I'm checking whether the first-layer sink is the only thing the later heads depend on, and I'm considering whether the L1 heads might show a recency pattern tied to the previous token rather than a fixed positional preference.

I'm checking whether the L1 and L2 heads show any positional structure beyond the early-position sink, by examining the P×P block of their QK matrices against the L0H3 subspace. The aim is to see if any of them carry rotation-like positional patterns, and how their magnitude compares to L0H3's.

I'm checking whether any of the later heads, especially the layer-1 ones, have a positional pattern like a previous-token head, beyond the early-position sink I already found.

