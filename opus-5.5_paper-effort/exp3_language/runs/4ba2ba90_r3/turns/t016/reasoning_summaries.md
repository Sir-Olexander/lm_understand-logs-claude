Checking whether L3H1 is an induction head: its score is high at +0.94, and its positional preference is +0.88, so it's probably copying. Next is L3H0.

Now I'm weighing L3H0 as a candidate: it scores around +0.33 on token match and +0.45 on previous-token match, so it attends to earlier positions with the same bigram context. That attention seems to land on the current token itself, which makes its copying role unclear unless its output carries information that the causal setup can't provide.

I'm now checking the L3H0 hypothesis against the induction definition: that definition expects a head to attend to the token right after an earlier match, but this head seems to attend to the matching position itself. The earlier L1 and L2 composition numbers for L3H0 come out negative, so I'll set aside the old positive reading and treat this head's role as uncertain for now.

I'm checking whether L3H1's key pulls from the L1 and L2 heads, since those show positive matches. That would suggest the earlier heads are gathering recent token context at each position, which L3H1 then matches against the query's current token.

I'm now looking at L1 heads: they strongly avoid attending to the same token as the current one, and their OV circuits seem to copy token identities from the positions they attend to rather than positional info. That makes me suspect they're local heads gathering recent characters, which L3H1's keys could then match against for longer-context induction.

Now I'm weighing whether the L1 heads are local, gathering the few preceding characters, or broader, pooling recent context. Either way L3H1's keys could match the query token against them, so the induction read still works. What's unclear is why a local head would avoid attending to the same token as the current one.

Now I'm looking at L2's heads: they barely attend to the same token and lean hard on positional preference, so they don't look like induction heads. I'll tentatively call them broad, though L2H1 seems to feed L3H1's keys.

I'm treating the L1 and L2 heads as mostly broad or local context gatherers, and I'll leave that label open for now. Next I'm trying a different angle: Python repeats exact text a lot, so the induction head should carry more of the load there than it would on English. I want to compare L3H1's output magnitude against the other L3 heads to see whether that holds.

Now I'm comparing how much energy L3H1 puts into the token dimensions against the other L3 heads, and it stands out as the main token-writer in that layer, so copying looks important here. I'm wondering whether newlines would be a strong signal in Python text, since they're followed by indentation.

Now I'm considering whether the model treats "space after space" as nearly impossible in English-like text, since runs of spaces get collapsed there, while Python indentation makes it common. That asymmetry could show up as a strong learned suppression of spaces following spaces, which would be a useful signal for telling the two corpora apart.

Now I'm weighing whether the L0 heads can even support counting spaces in an indentation run: there's only one sharp previous-token head and one local head, so that seems thin. Positional structure may matter less than position relative to the start of each line. I'm also wondering whether the broad heads are attending to newlines.

I'm now wondering whether the token identities could be inferred from how often each token is used: if the L1 heads have a default preference for certain key tokens, that preference might reflect corpus frequency, so newline could stand out in Python code in a way it doesn't in English.

I'm weighing a rough prior over the corpora: English around 0.5, Python around 0.3, and romaji around 0.2. Whether romaji really yields fewer dimensions is the part I'm least sure of, since the conversion was probably done with a Hepburn tool like cutlet and the loanword handling is unclear.

I'm considering whether the loanword spelling question matters here: if the corpus used foreign-spelling conversion, letters like l, q, x, and v would show up in katakana words such as "table" or "taxi", which would be a telltale sign of the source text.

I'm now checking whether the Aozora romaji data would still carry a distinct signal, since cutlet's foreign spelling could put l, q, x, and v into katakana loanwords like "table" or "taxi." That would make those letters rare but present, so they might give weak dims just like English's rare letters, which weakens the earlier argument. I'm also unsure whether cutlet outputs macrons by default.

Now I'm weighing whether the romaji text would even be distinguishable: Hepburn conversion tools like cutlet may keep English spellings for katakana loanwords, so l, q, x, and v would show up at low frequency, much like rare letters in English.

Now I'm weighing whether the romaji corpus is really distinguishable: if cutlet kept English spellings for katakana loanwords, then l, q, x, and v would appear at low frequency, so they'd act like rare letters in English and the token-count argument gets weaker.

Comparing the two corpora, I'd expect the Python model to rely on copy-style induction heads far more than the English one, since reusing identifiers is the main way to cut loss in code. The romaji case looks more local, built on consonant-vowel structure with fewer copy heads, and the shuffled versions should mostly fall back to broad attention.

Now I'm wondering whether the four broad L0 heads are really gathering bag-of-characters information, since their value reads positional signals rather than token identities. If that holds, they're likely encoding how far into the window each position sits, which later heads could use as a recency sink.

I'm wondering whether the induction head's strong positional preference means it favors recent matches, which might differ by corpus: copying from a few lines back in Python versus copying names from far back in English. I'm also considering whether attention sink patterns shift depending on whether windows start mid-line, since Python lines tend to be short.

I'm now looking at whether the output space can be untangled: rotating the 27 token dimensions so that MLP3's neuron outputs become sparse, which would give roughly one direction per token. I'd then need the input-side directions too, probably from the copy head's mapping of input tokens.

I'm now thinking about decoding token identities from the output space: the bigram-style structure can't be read off directly because the next-token prediction runs through nonlinear MLPs, so I'd have to lean on linear paths or on counting how many MLP3 neurons point at each token direction.

I'm wondering whether the rare-letter tokens in romaji, like l, q, x, and v from loanwords, could serve as the markers in the output space. Rather than reason from the top-level structure, I might look directly at how often the space token dominates across the MLP neurons.

Now I'm weighing the romaji case more concretely: the loanword letters l, q, x, and v would be extremely rare under foreign-spelling conversion, so q and x would barely register and l and v only weakly. That makes the weak output group in this model plausibly consistent with romaji rather than English.

I'm comparing how many token dimensions would clearly stand out for romaji, English, and Python: romaji's rare letters like q and x would sit close to the bulk, so it would show fewer distinct dims than the other two.

Now I'm checking whether the 27 token dimensions hold up against the romaji hypothesis, since the rare loanword letters should have produced fewer clearly separated dims. I'm weighing whether idx 28 is a real token direction or something special, which would leave 26 token dims with four weak ones, and I'm settling on English at about 0.5, Python around 0.3, and romaji 0.2.

Now I'm looking at whether Python's frequent identifier reuse would show up as several copy heads, the way it would in code. So far only L3H1 stands out clearly, which seems closer to the one or two copy heads typical of English.

I'm weighing romaji against English and Python by how many token dimensions stand out: romaji's CV structure might lean on prev-token and two-back heads, but the count of strong dims (about 21 versus 22) is too close to separate them cleanly.

I'm checking whether the L1 heads' same-token avoidance is really identity-like across all tokens, or whether it carries a class-level structure. I'll look at the eigen-spectrum of the symmetrized token-token block to see how many directions it actually uses.

I'm counting how many eigenvalues of the L1 heads' symmetrized token-to-token block are significantly negative, since roughly 27 would match the 28 tokens and roughly 23 would point toward romaji. I'll sum across heads to cut noise and use a 40-dimensional T.

