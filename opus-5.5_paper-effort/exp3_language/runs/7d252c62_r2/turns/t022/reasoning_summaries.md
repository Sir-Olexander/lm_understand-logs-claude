I'm checking two components with higher energy in the positional QK subspace and large key norms in the positional heads. That could mean they're special tokens the positional heads read on the key side, or that they encode position directly.

I'm testing whether components 2 and 22 are positional directions or tokens, since they're high in both Q and K. Their energy sits inside the token subspace (dirs 2–29) rather than the positional dirs 0–1 or 30+, and the ICA results suggest they're the space and newline tokens. I'm still weighing a romaji case with 24 tokens, which would leave four extra dimensions.

The rare-letter idea is the next thing I'm weighing: q, x, z, and j show up least in English, while Python code leans on letters like f, d, and k in different proportions. Since the ICA gave me 26 sparse letter-like components plus the two specials, I'm checking whether those rarer letters are still cleanly separated despite their low frequency, because that's what a letter-frequency comparison would depend on.

I'm weighing whether the rarer letters like q, z, x, and j can be told apart by how often they show up in English versus Python code, since Python leans on letters like k, v, and w more. I'm also wondering whether the number of dedicated MLP0 neurons per component tracks how frequent a token is, though that isn't clear yet. One option I'm considering is checking induction diagonals for each token.

I'm checking whether comp 4 behaves like a frequent letter, given its low induction-diagonal values across L2H1 and its odd L1 and L0H0 signatures. Induction would matter less for a common letter like 'e', which might explain why those heads ignore it. I'm also wondering whether comp 4 is the space token instead, with 2 and 22 as the newline pair.

Now I'm looking at the L1 heads' key biases: comp 4 and comp 2 get pushed down across most of them, while comp 22 gets boosted in several heads, which would fit newline tokens acting as attractors.

I'm weighing whether components 2 and 22 are space and newline, with 4 perhaps being the frequent letter 'e'. L1 heads would then steer away from spaces and common letters while leaning toward newline keys as line-start anchors, though that pattern wouldn't matter much in English where newlines are rare.

I'm weighing whether newline tokens could be told apart from spaces by how they behave around each other, since newlines shouldn't usually be followed by a space in the same way. I'm thinking about using the prev-token heads' sign patterns to tell which of components 2 and 22 is the newline.

I'm weighing whether induction heads treat token 2 differently from token 22: token 2 barely matches in the induction diagonals, while 22 matches moderately. If 2 is the space token, then matching on "previous token is a space" wouldn't single anything out, since spaces are everywhere. If 22 is the newline, then matching a newline query against keys whose previous token is a newline could recover what followed newlines earlier, such as indentation.

I'm thinking about whether token 2 could be the space token, which would mean a query on a space attends to earlier spaces rather than the preceding letter. Spaces inside Python indentation would look like that, while in English a space follows a word's last letter, so the pattern would separate the two.

I'm weighing whether the vowel-consonant pattern in English, or the letter-pair structure in the L1 heads, could separate the two corpora, since q-u is such a strong English bigram.

I'm now weighing whether vowel and consonant alternation in romaji and English could be picked out in the 26 letter components, and I'm considering the Python case too. I'm also thinking about how indentation could be tracked by counting spaces since the last newline.

I'm weighing whether token 2 is a newline or a space, since L0H0 and L0H3 read it strongly as a landmark for the last line break. If 2 is the newline, those heads would be tracking line starts and column position, which fits Python structure. If 2 is the space, they'd be marking word starts instead, and I'd need the L1 avoidance pattern on tokens 2 and 4 to decide between the two.

I'm settling on token 2 as the space and token 22 as the newline, since L1 heads avoid spaces and favor newline keys. That would make L0H0 and L0H3 track the last word start, with L0H3 pushed negative by newlines. Token 4 is still unclear: it's avoided by the L1 heads and gets a positive L0H0 key value, so it might be a frequent letter that acts like a word boundary.

Now I'm weighing whether the 2-as-space, 22-as-newline reading holds up, since newline seems to get as much weight as space in the positional heads. That looks odd for a rare token, though I'm wondering if the newline might just be treated as a strong paragraph boundary.

I'm checking whether comp 22 really is the newline token, since the newline structure in the positional heads is roughly the same size as the space structure. That would be surprising for a rare token in English, but it fits Python, where newlines and indentation are common. The space-versus-newline key contrast would then matter a lot for telling line starts apart from indentation.

Comp 22 looks like the newline token rather than a letter like 'e': it has the most strong MLP0 neurons, the largest positional-head key norm, and a strong value readout in L0H4, none of which a single letter would produce. If newline matters about as much as space in these heads, that points toward Python over English, though I want to check the English Wikipedia case more carefully.

Now I'm weighing the Wikipedia case against Python: article text has fewer line breaks, maybe around half a percent of characters, while Python code has far more, roughly three percent. I'm also considering indentation, where a newline is followed by a run of spaces that the positional heads would see as a line start followed by repeated spaces.

Now I'm checking whether comp 4 is special in Python, perhaps 's' from "self" or 'e', though I can't pin it down. I'm also considering whether space-after-space patterns in indentation would be learned differently from space-after-letter, since English rarely has that adjacency.

I'm checking whether the positional heads track the last space or the last newline, since L0H0's space response is strong and I want to see if its newline counterpart is just as strong. If the induction and positional patterns look the same for both, the signal probably doesn't settle the question on its own.

I'm now looking at L0H4, which reads both space and newline almost equally, and wondering why it would weight newlines so heavily if they're rare in English. That might suggest the data has many line breaks, which would favor Python, where line density varies with code structure.

I'm weighing Python against English at roughly 55-60% versus 30-35%, with romaji around 10%. Newline-preferring keys in the L1 heads look like they could track line starts, which would fit Python's indentation runs and make the newline-space contrast matter.

I'm checking whether the L1 heads' preference for newline keys is mostly a Z-direction effect or really a token effect. Their key bias is dominated by the sink features, so the token part is smaller than it first looks, and I want to measure how big the newline contribution is in logit terms.

Now I'm weighing whether the L1 heads' pull toward newline keys can be read as a clean token effect, and I'm also checking whether romaji, using all 26 letters, could be the corpus given how many sparse components came out.

I'm wondering whether a rare token like 'l', 'q', 'x', or 'v' could still get dedicated sparse neurons if it appears often enough in the training data. The count of letters may not settle whether the corpus is romaji, so I'm checking how strongly L0H5's value readout depends on each component.

Even if the rare letters only get small singular values, that wouldn't mean they lack dedicated directions. The singular values of W_v.0 vary only about 2x while letter frequencies differ roughly 100x, so frequency doesn't map cleanly onto strength. A letter at around 0.005% could still get a usable direction, which weakens any inference from letter coverage alone.

I'm weighing whether romaji could still be the corpus, since 28 sparse components don't fully rule it out, and romanized Japanese would have high vowel frequency and a different newline rate. Romanization roughly doubles the character count of a paragraph, so newlines in romaji text might show up around one to two percent of characters.

I'm estimating that romaji would likely lack a few letters like l, q, x, and v, since Hepburn romanization rarely uses them, though c and f still appear often through "ch" and "fu".

I'm comparing the vowel structure next: romaji would put the five vowels at about half of all letters with near-regular consonant-vowel alternation, which should show up as a distinct cluster in the token basis, while English vowels cluster more weakly.

I'm checking whether the L0H5 and L0H1 key-side loadings on the sparse components point to particular letter groups. Then I'm planning to look at the MLP0 neurons in the token basis for co-activation patterns, since clusters there could show whether the vowels stand apart from the consonants, which would matter for romaji.

I'm weighing whether the vowels would form a tighter cluster in romaji than the consonants do, and whether that tells me anything. Python's frequent letters like 's', 'e', 'l', and 'f' could be another cluster to look for. I'm planning to check line-level structure next, since short Python lines would fill a 128-token window with several line starts.

I'm now checking whether the newline-to-space pattern in indentation shows up in the induction heads, since copying a previous line's indentation would need newline keys matched to newline queries. I'm also weighing the corpus size of about 320 million characters against the three-epoch training budget, and whether that changes how dedicated the sparse neurons would be.

I'm looking at whether the positional heads' space-to-space attention could be tracking the run of spaces within indentation, similar to how a head might count recent spaces since the last newline. For English, the same term would instead point back to the previous word boundary.

Now I'm looking at L1 heads, which seem to pull toward newline keys, and wondering if one of them acts as a line-start marker that tracks column position for indentation. I'm weighing that against English, where newlines rarely appear in the window, and leaning slightly toward Python while I check whether component 22 is really a letter.

I'm comparing the key norms of components 2, 22, and 4 in the positional heads, since those three seem to be the tokens whose identity changes how the positional heads read keys. Space and newline are the obvious structural candidates, so I'm wondering what the third one is for Python or English, and I'm checking whether component 4 behaves differently in the L1 heads.

I'm checking whether comp 4 could be 'e', since it's the most frequent letter, but its low induction score and the L1 heads' avoidance of it don't obviously fit that. I'm also wondering if it's a mixed direction rather than a clean token.

I'm checking the induction-head diagonal scores next: comp 2 sits near 0.3 while comp 22 is around 1.1 to 1.7. Matching on a newline could help with repeated indentation in Python, but matching on a space alone says little in English, since the previous word would matter more than the space itself.

I'm settling that component 2 is the space token, since it has the lowest copying and induction scores, which leaves component 22 as the newline. Newline has the largest positional-key norm and the strongest value readout, so I'm asking whether positional heads might avoid attending across newlines, as a paragraph-break signal, even in English.

I'm weighing whether the newline structure would be comparable in Python, since it occurs about ten times more often than in English. But with Adam, the update depends on gradients accumulated across batches, and newlines appear in every batch in both corpora, so frequency alone may not settle it.

I'm weighing whether the uniform head L0H4 would read space and newline densities differently across the two corpora: in Python those densities track indentation depth and line length, while in English they stay roughly constant, so the head would carry little information there and mostly track position.

I'm now wondering whether comps 2 and 22 look large in the positional heads only because they overlap with the positional subspace P, which would make them an artifact rather than special tokens. The same overlap could also inflate their key norms there, so the evidence for them being space and newline is weaker than I'd assumed.

I'm checking whether comps 2 and 22 are real token directions or just positional mixtures, since the positional QK subspace already includes special-token directions, which makes the overlap circular. A cleaner test is whether they show up in L0H4's top value directions, which are mostly positional; their readout there is only partial, so it doesn't settle it.

I'm reasoning that comps 2 and 22 are token directions, not positional mixtures, though the P-overlap makes that circular. I'm weighing a cleaner check: whether their readout in L0H4's top value directions is strong enough to settle it, and how that would shift the Python-versus-English reading.

Weighing newline frequency against the English corpus, I'm thinking that a period-4 indentation pattern in Python could show up as periodic features in the positional heads, so I want to test whether the L0 heads track column position since the last newline.

I'm checking whether the prev-token heads L0H1 and L0H5 split space and newline with opposite signs, and whether a head like L2H2 copies the previous token's identity.

I'm weighing whether the induction heads and "self"-heavy keyword statistics would show a strong copying signal in a Python model, and I'm thinking about whether I could check vowel structure to tell English from code.

I'm wondering whether the letter-frequency pattern in the token-level interactions could tell English apart from code. Since the L1 heads' QK is in token basis, I could look for letter pairs that are fixed in English, like 'q' followed by 'u', but attention alone doesn't reveal that, so this route is hard to pin down.

I'm now trying to read the bigram structure from the MLP0 neurons: for each neuron, comparing its response to the current token against its response to the previous token, to see whether sparse peaks reveal which letter pairs it detects.

I'm weighing whether a head that attends to the most recent non-space token and copies its identity could detect being inside a line's leading indentation, which would favor a Python reading over English.

I'm checking whether L0H5's sign flips the reading: if its key preference is negative, it would attend to the most recent non-space token and favor newlines, which would signal indentation; if positive, it would attend to the last space and avoid newlines.

I'm testing whether the sign of the query-side positional vector flips the reading of L0H5: if it's positive, the head favors the last space and avoids newlines, but if it's negative, it favors newlines and would signal indentation. I can't pin down that sign without the mean positional embedding, so I'm checking whether the constant bias term gives a usable query direction instead.

I'm trying to tell which sign the positional query components take, since L0H1 and L0H5 prefer opposite tokens: one attends to space keys and avoids newlines, the other the reverse. I can't pin that sign down without the mean positional embedding, so I'm wondering whether a shared constant component across positions would fix it.

I'm trying to pin down the sign of the positional query components, since L0H1 and L0H5 prefer opposite tokens. Without the mean positional embedding I can't fix it directly, so I'm considering whether a shared constant component across positions would settle it. Downstream, if L0H5 picks out the previous non-space token, induction heads would effectively skip over spaces when matching.

I'm now weighing whether L0H5's value vectors barely read the space token, which would mean the head seldom attends to spaces, and so it would behave as a "previous non-space character" detector. That would give an indentation signal in Python (finding the newline) and point at the last letter of the prior word in English.

I'm checking whether L0H1's weak reading of space tokens means it isn't copying space identity, leaving that job to the landmark heads that track distance since the last space. Then I'll test whether a romaji vowel-after-consonant pattern could be picked up from the same heads.

I'm checking whether the rare-letter components 14 and 15 fit a romaji corpus better than English, since romaji would leave l, q, x, and v with few dedicated neurons, and I'm comparing their kurtosis and induction scores against that expectation.

I'm checking whether the weaker value readout on components 14, 15, and 21 fits romaji's rare letters, comparing their norms to the typical level and to the bulk singular values in the value weights.

I'm now comparing the sizes of the value-direction weights for each letter component, expecting that if the corpus were romaji the rarest letters would show near-noise strengths, while a smooth falloff would point toward English or Python.

I'm checking the letter components' value strengths against each other, and they all look comparably copied, so romaji seems unlikely. Next I want to look at component 4, whose positional key norms are high, to see whether it behaves like a structural token.

I'm now looking at component 4, which has high positional key norms but is strongly avoided as a key by the L1 heads, with low induction scores. I'm wondering whether it's a frequent letter like 'e', though that doesn't obviously explain its treatment by the positional heads.

I'm now checking whether comp 4 is a vowel by looking at how the letter components cluster in the prev-token heads' second key contrast, where it scores strongly positive while comps 11, 25, 9, and 15 score negative.

I'm checking whether the letter components split into vowels and consonants along the PT2 contrast, with component 4 and a handful of others on the positive side and the rest negative. If the positive group is about six components, it could be the vowels, with component 4 standing out as the strongest one.

I'm weighing a vowel-versus-consonant split along the PT2 contrast against romaji and English, but that doesn't separate English from Python. Single-letter identifiers in Python code could be a better signal, so I'm checking how often short tokens like those appear.

I'm weighing Python over English for now, since newlines would give the L1 heads a strong sink signal to latch onto, and the newline preference in those heads lines up with that.

I'm now considering the counter-hypothesis that the corpus is English Wikipedia, where newlines would be rare and act as context boundaries that the L1 heads might use as sinks. I'm also weighing what structure Python would add that English lacks: leading runs of spaces after a newline, which would mean spaces can follow either a newline or another space.

I'm checking how the L1 heads might encode the position of the last newline, since indentation counting would need that, and I'm weighing whether their value readout of the positional embedding is strong enough to carry it.

I'm wondering whether the uniform head L0H4 averages in newline density, since a rare newline would barely register there. I can't check that without a forward pass, so the indentation-counting mechanism stays unconfirmed.

I'm now wondering whether the L0 heads' positional query shift differs for space and newline tokens, which would let a head attend to the previous line's start when the query is a newline, a pattern that would copy indentation.

I'm now planning to print the TP query-token functional for L0H1 and L0H5 in the token basis, to see which tokens shift the positional preference, and I'll compare the magnitude for newline queries against space queries.

