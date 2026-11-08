The first detector clusters I checked mostly show a negative weight on the current token when the previous token differs, which looks more like repeat suppression than a useful signal, so they don't separate English from Python well. I'm stepping back to look for other features that might tell the two apart. One candidate is line length: Python lines tend to be short, so newlines come often and would show up within the 128-token window.

I'm weighing whether the line position itself could carry the signal. Indentation would make Python lines repeat the same leading spaces, so a head that matches on column position within the line might pick up code structure that English lacks. I'd need a "column" feature built from the last newline's position, which the earliest attention layers could plausibly compute.

I'm considering whether the "column" idea could be computed by early attention heads and whether later heads could match on it, though that gets complicated fast. I'm also weighing the difference between attention-heavy copying in code and MLP-driven n-gram recall in prose, and whether per-layer weight norms could show it, though I don't have a reference to compare them against.

Checking the attention patterns for recency bias, I'm seeing L2H1 with a weak position preference, so it looks long-range, while the L3 heads don't show much key bias either. I'm also thinking about whether newlines followed by indentation make the next token predictable from the previous line.

I'm looking at whether the copy heads on newlines would matter more in Python, since indentation makes the next line start predictable from the previous one. Spaces after a newline are harder, because matching on space runs alone is ambiguous without some sense of column position. So I'm checking the heads that read from the early positional layers to see if they encode that column information.

I'm still weighing whether a column-matching copy head could work here, using the positional outputs of the early layers. Then I'm wondering whether the newline-related heads in layer 0 would matter at all, since newlines are rare in English but central to Python structure. Averaging the positions of all spaces seems nearly useless as a signal.

I'm checking whether L0H0's selectivity is meaningfully different from the uniform positional head L0H4; its QK norm is about 13 times larger, but still weak next to the stronger heads. So I'm treating it as a broad head with mild token weighting, and I'm thinking about whether recovering token embeddings would give a better handle on it.

I'm checking whether the copy heads' query-key matrix for matching tokens looks different for spaces, since in Python spaces repeat in runs while in English they almost never do.

I'm weighing whether a positional head at offsets 3 or 4 might capture indentation structure, and I'm going to check whether L0H2 and L0H3 behave like the offset-2 heads I already looked at. I'm also considering the L1 heads that read the previous tokens, which would give the standard local n-gram signal.

I'm wondering whether the positional structure could reveal how many distinct relative offsets the model resolves sharply, since Python indentation might need more positional dimensions than prose does. I'm also considering a romaji angle: its strict consonant-vowel structure would make next-character class easy to predict from the previous token, though repeated particles are too short for copying to help much.

I'm considering whether the copy-heavy attention structure favors Python over English, since indentation and repeated identifiers would make copying more useful there. I'm also checking how the prev-token head and MLP0 might handle the newline-then-space bigram, which would be a strong structural cue in code but rare in prose.

I'm estimating the scale of the L3H5 induction logits by comparing their composed matrix norm against the positional logits of the L0H5 prev-token head, since both depend on similarly normalized inputs. The token embedding norms within the LayerNorm output are the unknown I still need to pin down.

I'm looking at a broad attention head in layer 3 that spreads its weight across the whole context and writes a few dimensions very strongly, and I'm wondering whether in Python that kind of averaging could track the indentation level of the surrounding lines.

Now I'm checking whether the newline-driven positional signal survives in the L1 heads. The idea is that a head might attend only within the current line by matching on whether the L0H0 output agrees between query and key, so I'm measuring the strength of that same-line matching term across layers 1 through 3.

I'm checking whether O0 and O4 really occupy the same output plane, since that decides whether the L0H0 same-line signal can be separated from L0H4. The earlier input cosines were high for those heads, so I'm wondering if their outputs overlap too, and I'll verify that before going further. Then I'm considering whether a benchmark-style answer would be expected here.

I'm weighing which corpus the attention structure points to: Python would show strong induction heads and broad heads tracking indentation, while romaji would lean on local character patterns. The early layers look local and the layer 2–3 heads look copy-oriented, and I suspect Python would produce the most copying of the three.

I'm counting the rare tokens in the spectrum: about six weak dimensions, which would fit English's rare letters like v, k, j, x, q, z plus newline. Python would have fewer rare tokens since k, v, x, and y show up often in code, so the weak-dimension count points toward English.

I'm checking whether the spread of singular values in the token subspace shows a split: about 21 dimensions at or above 0.54 and about 6 between 0.28 and 0.45. That six-dimension tail could match English's rare letters, so I want to compare it against English letter frequencies.

I'm comparing English letter frequencies against the singular value spectrum: space plus the twenty most common letters would make up the 21 strong dimensions, and the rare letters v, k, j, x, q, z plus newline would account for the six weaker ones. That split fits English well.

I'm noticing the 21 strong and 6 weak dimensions line up neatly with English letter frequencies, if space and the common letters count as strong and v, k, j, x, q, z plus newline count as weak. Now I'm sketching rough Python letter frequencies from memory, where e, s, t, and r lead and f gets a boost from "self", "def", and "if".

I'm estimating how the Python letter frequencies would split into strong and weak dimensions. Space and newline are strong, along with roughly the top twenty letters, and k, v, w, j, q, and z fall into the weak group. That gives about the same 22-strong, 6-weak split as English, so the spectrum count alone doesn't settle it.

I'm wondering whether the spectrum's gap between the 21st and 22nd values means anything, but it looks too modest to settle the corpus question. Since the frequency split doesn't separate English from Python, I'm going to look for a different signal, maybe in how the MLP0 neuron clusters are sized.

Now I'm wondering whether the anti-self heads in layer 1 could tell Python apart, since at a space they'd look back to the most recent non-space token and reveal whether it's a newline, meaning the model is still in the indentation region. That kind of signal would matter much more in code than in English, where the most recent non-space token is just the end of the previous word.

I'm wondering whether the spectrum can't settle the corpus question by itself, so I'm trying a different angle. In Python, the space token plays two roles, a separator and an indentation marker, so I'd expect one dominant eigendirection for it. The L1H1 eigenvalues look fairly spread, with no single dominant one, which is weak evidence in either direction.

I'm thinking about what the newline token does in English Wikipedia text: if articles are split into paragraphs and section headings sit on their own lines, newlines would come up roughly once every few hundred characters.

Now I'm wondering whether the induction heads treat the space token differently from other tokens, since that might reveal whether the corpus is code. I'm also considering whether a head could attend back roughly one line, about 30 to 40 characters, to approximate the same column on the previous line.

Next I'm checking whether induction matching on space or newline would look different between Python and English, since spaces are everywhere in indentation for code while newlines would be the key cue for copying indentation. Then I'm looking at how strongly the rare-token dimensions show up in the induction matrix, since copying rare letters like q, z, x, or j might carry the most weight there.

I'm trying to estimate token frequencies more concretely from how the prev-token head's OV should scale with how often each token appears. With Adam, though, the per-parameter normalization probably weakens that frequency dependence, and rare-token embedding rows only get updated when those tokens show up.

I'm working through whether Adam's per-parameter normalization makes rare tokens learn about as fast as frequent ones, since with enough characters per batch even a rare letter like z shows up every batch. That would weaken the token-count argument, especially if romaji text contains some Latin letters like l, q, and x that appear only occasionally.

I'm weighing whether Aozora texts would contain enough Latin letters to matter. Words like "Faust" or "ABC" might show up, and after romanization they'd stay in the stream, so the rare tokens may get some training, though far less than common ones.

I'm working out how much weight rare romaji letters like l, q, v, and x would get in the shared matrices, since those directions should scale with frequency even under Adam. At roughly 0.01% they'd be about an order of magnitude below English z and q at 0.07% and 0.1%, which is at odds with the 0.28 to 0.45 rare-token dimensions I saw for English. I'm now checking how the letter c, mostly appearing in "ch", would compare.

I'm checking the rare-token estimates again: the romaji letters l, q, v, and x at about 0.01% should sit at the floor, and that matches the 27 dimensions above it, so the English/Python question still needs its own frequency count. I'm starting to list the English rare letters, v at about 1.0% and k at about 0.8%, to see how many weak dimensions they'd account for.

Working out how rare tokens like j, q, and z would look in Python, with their frequencies sitting roughly 0.2 to 0.4 percent, compared with about 0.1 to 0.3 percent in English. I'm wondering if the singular values of the weak dimensions, which look fairly close together, can be calibrated against frequency at all, and I'm planning to check the refined spectrum through the MLP0 weights.

Still weighing the token-spectrum angle: the frequency decay across English letters is steep, roughly 180× from "e" down to "z", so the spectrum might be sized differently under Python's distribution. I'm not sure that settles anything yet, so I'm turning to structural patterns in the attention heads instead.

I'm weighing whether attention-versus-MLP output norms could separate the corpora, since the MLP projection norms look larger than the attention output norms, but I have no reference point to read that against yet. For now I'm leaning toward a rough split of about 0.45 Python and 0.42 English, with romaji lower at around 0.13.

I'm checking whether the attention heads show a positional pattern tied to four-space indentation, and wondering if an offset-4 head would pick up the column structure of Python code. I'm also comparing how many distinct positional offsets the layer-0 heads target, since English models often combine previous-token, two-back, and local recency heads.

I'm checking whether L0H2 prefers a whole class of tokens or just one, since that would show whether it's tracking the most recent space or a broader set of informative characters.

Now I'm weighing whether MLP0 neurons that combine a space with a following space would be trained at all, since English text rarely has that bigram, so I'm planning to compare each neuron's cur-token and prev-token functionality to see if that gap shows up.

Now I'm weighing the space-space bigram as a possible test: in English the pairs of spaces, or newline followed by space, should basically never occur, while Python would show them often. Without labels I can't check that directly, so I'm setting it aside for now and turning to the structure of line starts in Python.

I'm checking whether the head at layer 0 that looks far back is keyed to a single token, by dotting its key direction against each cluster centroid from the first MLP. If one centroid clearly dominates, that would point to a specific token rather than a broad class.

I'm checking whether the two largest clusters, the ones with 35 and 29 members, are just the most frequent tokens like space, since that would explain their dominance.

