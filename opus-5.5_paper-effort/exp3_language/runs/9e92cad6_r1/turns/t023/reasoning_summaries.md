I'm comparing cosine values at different top-k settings for the 23rd through 25th ranked items. The values for the 23rd and 24th items grow as k increases, though part of that growth looks like random-level noise. The knee still sits between indices 22 and 23.

I'm checking whether the cosine values for the third-ranked token dimension reproduce the knee near index 22. My working estimate is that the token dimension count is around 22 to 24. I'm now weighing whether an English text of 28 tokens would show 27 to 28 dimensions or lose the rare ones, so I'm going through English letter frequencies to see how many symbols stay above a 0.5% threshold.

I'm weighing whether rare letters in English would fall off a 0.5% threshold, which would make the 23-24 token count look the same as romaji. Python identifiers might keep more symbols above that cutoff, so a code sample could plausibly show around 28.

I'm now asking whether the rare letters would really fall below the 0.5% cutoff, and whether that would leave English with the same count as romaji. Python identifiers might keep more symbols above that line, which would argue against it, but I'm not sure the normalization step lets me count them that way.

Python predicts around 27-28 dimensions, but I'm seeing 23-24, so Python looks less likely unless the knee isn't really about vocabulary size. That leaves English versus romaji, which both come out near 24 frequent tokens once the rarest letters fall below the cutoff. I need a different check to tell them apart.

I'm now looking at whether the attention heads in the first layer show a vowel-consonant pattern, where a vowel position attends back to the preceding consonant. If that were the main structure, the symmetric part of the same-token matrix would have one strongly negative eigenvalue, and I'd check for that.

I'm weighing whether the rare letters would clear the frequency cutoff differently in English than in romaji, and whether a newline frequency difference would separate them. Next I'm thinking about how word-level spacing might differ between the two.

I'm checking whether the logit-side dimensions fit a 27-token English vocabulary or a romaji one. The strong dimensions cover about 21 tokens, with a few medium ones after that, and I'm wondering whether the rarer letters like x, j, q, and z would show up as weaker separate dimensions.

I'm checking whether the logit dimensions 25 through 28 are really rare-letter outputs, since the broad head L3H4 writes into them strongly, which doesn't fit a plain rare-symbol story. Maybe it's a bag-of-context unigram head boosting tokens already present in the context.

I'm wondering whether L3H4 is a broad "unigram cache" head that boosts rare letters already present in the context, since its output lands on the dimensions 25 through 28 that I'd otherwise read as rare-symbol logits. That would explain why a rare-letter story and a copy story both fit the same dimensions. I'm going to check its OV output against the dimensions to see which reading holds.

I'm checking whether L3H4's value vectors pick up on anything in the earlier MLP or attention outputs, since they don't seem to read raw token embeddings. I also want to see whether its broad output points more toward structural features like position or line length, which would fit the logit dimensions it writes into.

I'm looking at whether the sparse directions 22 and 25 in the final layer's output weights could be rare-token logits, since only a few neurons write into them.

I'm weighing whether dimensions 22 and 25 in the output weights could be rare-token logits, and I'm reconsidering a way to link directions to tokens through the induction copy map.

I'm weighing which weight features would reveal the corpus: Python might show newline and indentation heads plus heavy induction, romaji a smaller active token set with a strong previous-token head, and English something in between. Given the roughly 23 to 24 dimension estimate, romaji looks plausible, and I want to find evidence that about four tokens are effectively unused.

I'm weighing whether the rare letters in English would survive a 0.5% frequency cutoff, and I'm also considering that romaji would hardly contain l, q, v, or x at all since the source text is Japanese. That would make the rare-letter dimensions a clear difference between the two corpora, so I'll check which dimensions go quiet.

I'm now checking whether the shared structure survives once the common subspace is removed, by projecting out the shared directions and seeing if any extra overlap remains beyond random.

I'm now removing the attention-output directions of the layer-0 heads from what MLP0 reads, so the remaining input-side spectrum should show the token dimensions without that interference.

