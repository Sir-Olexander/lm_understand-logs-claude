I'm noticing that every layer-1 head strongly avoids attending to positions holding the query's own token, with large-norm negative scores. Layer-2 heads and the previous-previous-token pattern also come out negative. Layer-3 head 3 stands out, scoring high on both induction and previous-previous-token matching.

I'm working through the L3H3 numbers and trying to square its induction and previous-previous signals with what L0H3 would be doing. If L0H3 isn't just the previous token but a short local window of recent characters, then matching on that bag of context would explain both positive scores at once.

I'm going back over why the layer-1 heads would avoid same-character positions, especially in shuffled text where the next character should come from what's left in the line. Uniform attention over the line would just track letter frequencies, so the anti-same-token pattern doesn't obviously help there. I'm also wondering whether the trace-based approximation of same-token scores holds, since it only works if the embeddings are roughly isotropic within that subspace.

I'm weighing whether the shuffled-line case really explains the anti-same-token pattern. The predicted next character should scale with how many of each letter remain in the line, so seeing a character in the prefix pushes its prediction down while also suggesting the line holds more of it. That tension makes the simple frequency story look shaky.

I'm checking whether L0H3 could be a short local window rather than strictly the previous token, since that would make the anti-same-token pattern in layer 1 harder to explain by frequency alone. The shuffled-line picture still doesn't cleanly account for avoiding same-character positions, so I'm looking at the QK and OV structure of the layer-1 heads to see what they actually read.

I'm going back to the question of whether L0 has several positional heads at different offsets, since char-level n-gram information would usually need that.

I'm wondering whether L0H3 might be a short local window over several recent characters, which would explain positive scores on both induction and previous-previous matching in L3H3. A uniform attention window over the current line would also give a bag-of-recent-characters signal that could matter in shuffled text.

I'm now asking whether L3H3's induction and previous-previous signals could both come from a local-window bag in L0H3, since a standard induction head would instead match the prior token and the one before it. That would make its positive scores consistent with recent-context similarity rather than a strict offset match.

I'm working out whether L0H3 is really a strict previous-token head or a small local window. If it mixes positions i-1 and i, the query-side and key-side outputs would both carry the current character, which could make L3H3's induction and previous-previous scores positive at once. That would leave the matching ambiguous, so I want cleaner evidence about the positional structure before settling on either reading. Next I'll look at L0H0's positional QK, whose eigenvalues are both positive.

I'm considering whether the layer-1 heads that read a shared positional subspace might be outputting the position of the most recent newline. If so, later heads could favor keys from the current line, which would make sense for short lines like python or romaji but matter less for long english paragraphs.

I'm sizing up how strong the Fout key preference is, by estimating the score contribution from the LayerNorm-normalized L0 output projected onto that direction. The open question is what dominates the layer-1 residual stream, since that sets how much this query-independent bias actually matters.

The layer-1 keys seem to carry a strong query-independent preference for positions with a high Fout component, which would act like a segment mask. I'm checking whether the L0H1/2/4/5 query-key directions actually read the token or position subspaces I've been using, since they don't show much overlap with those.

I'm checking whether L0H1, L0H2, L0H4 and L0H5 read the T and P subspaces at all, by extending those subspaces to more dimensions and seeing whether the overlap grows. I'm also considering L0 neurons as another route into the positional structure. Stepping back, the question is which corpus and ordering produce these heads: prev-token and induction-style heads usually mark as-is text, while broad heads point to shuffled input. The layer-1 segment-like behavior and L0H3's local positional role both need to be tested against that split, and python with its newlines and indentation would look different from long English lines.

I'm weighing whether a segment-like key preference in layer 1 could come from the newline position, and whether that would matter more for short python or romaji lines than for long english paragraphs. I'll also check the layer-0 neurons for positional structure, though I doubt that route is recoverable.

I'm estimating the positional embedding curve from L0H3's QK structure, assuming it acts as a pure offset-δ head. The angles of its complex eigenpairs in P, roughly 35 to 61 degrees, look close to a common value, which would fit frequencies ω_k·δ being about 45 degrees for every plane.

I'm now checking whether the angles of those complex eigenpairs in P, all clustering around 45 degrees, can be read as frequency-times-offset values. The eigen-angles might not map cleanly onto the positional frequencies if the query and key read different coordinates, so I'm considering a different route: looking at the 2-d Fin subspace inside P to see whether it captures a dominant positional trend.

I'm checking whether Fin fits inside the rotation planes of L0H3's A matrix, which would let me read positions as points on a circle. I'm also looking at L0H0 next, since its QK has real positive eigenvalues and no rotation, so if its query and key both sit in the Fin direction it could be a recency head favoring late positions.

I'm weighing whether L3H3's induction behavior could be mirrored across corpora by checking how python's repeated identifiers and indentation runs compare against English and romaji text, since shuffling would scramble the leading spaces.

I'm checking whether L3H3's matching holds up on its own, and I'm also looking at L3H4, which has a tiny QK but a large OV, likely a broad uniform head. Shuffled text would need a bag-of-letters signal for the current line, so I'm wondering whether a segment mask or L0H3's local window would supply it.

Anti-same-token could just be a way of counting characters in the current line, and that may not matter here. Maybe the score for matching the query's own token is large and negative, around -6 under a rough estimate, which would mean strong avoidance.

I'm wondering whether anti-same-token behavior combined with a recency preference could act like a previous-token head, since the self position always holds the query's own token and so gets suppressed.

I'm wondering whether the Fout direction could be a recency signal, rising with position, so that a recency preference combined with suppressing the query's own token would behave like a previous-token head. That would mean the six layer-1 heads might all be local or previous-token types, though with different strengths. Alternatively, in shuffled text the same bias could mean attending within the current line, but then avoiding same-character positions would throw away the counts the head needs, so that reading looks less likely.

I'm checking whether the layer-0 heads H1, H2, H4 and H5 act as a single query-independent bias that favors keys with a particular token feature, since their negative rank-1 scores would flip depending on whether the query sits on the same side of that feature.

I'm projecting Fin onto the eigen-planes of A3 to see whether it lines up with the rotation structure there. I'll also check whether L0H0's top QK directions point toward Fin, and whether P carries word-boundary features instead of positional ones.

Next I'm looking at whether the token subspace has the rank I'd expect: a prev-token head that copies token identity should show roughly one fewer dimension than the number of tokens it uses, so I'm checking the singular values of L0H3's OV input against that.

I'm checking whether the token subspace has the rank a previous-token head would need, but the singular values don't show a clean cutoff. The rare characters probably contribute little, so I'll look at the residual dimensions where the layer-0 norm weights are large instead.

I'm looking at whether ln1_w.0 shrinking from 1 toward around 0.24 is a sign that the model is downweighting the raw inputs to layer 0, since the layer-norm gains are not subject to weight decay and so can drift freely under gradient.

I'm now considering whether the corpus differences show up in line structure: English paragraphs are long with few newlines in a 128-character window, Python lines are short, and romaji paragraphs run long because kana expand to multiple letters.

I'm asking whether the layer-0 heads' key-preference vectors could reveal a newline direction, but the vector for H1 is too small to tell, so I'm thinking about looking at the L0 MLP neurons instead to see how they react in shuffled text.

I'm testing whether the token subspace T is really separate from the positional subspace P, and the small overlap suggests it is. L0H3 reads almost nothing from T in its query and key, roughly what random directions would give, so I'm checking the P dimensions more carefully.

I'm reading L0H3 as a purely positional head: its query and key scores depend only on the positional subspace, so it copies token identity from a fixed relative offset, much like a previous-token head. Its rotation-based scores make it non-symmetric, so it never attends to itself.

I'm reasoning that the late-position preference probably comes from the Fout direction encoding the position of the most recent special token, so keys in the current segment all share the same value, the largest one. That would make a segment-start step function look like a recency bias.

I'm wondering whether every head in layer 1 is simply attending uniformly over the current segment, whether that's a word or a line, while avoiding its own character. For shuffled text the bag of the current line would carry most of the signal, which would explain why so many heads might be doing it in different ways.

I'm wondering whether L0H5 attends to newline or space tokens, but I can't tell that directly without token identities. In shuffled text the spaces are scattered randomly, so the segment boundary that matters is probably the newline. Still, L0H3 being a purely positional copy-the-token head seems odd here: within a shuffled line its previous character is just a random sample of the line's letters, no more useful than any other member of the bag, and the L1 heads reading its output would only be repeating that same bag information shifted by one.

I'm checking whether L0H3's offset could be a local window that detects line boundaries, such as whether the character one step back is a newline. To test order-sensitivity, I'd look at whether the layer-1 heads match on the previous character's token-identity signal against the key's token content, and whether the sign of that match changes.

I'm weighing whether L0H3 attends to just the previous position or to a small local window of offsets. Under the window reading, L3H3's induction match would be a fuzzy one that counts the query's recent characters against the keys' recent ones.

I'm checking whether L3H3's OV reads the key's own token, which would fit an induction-style copy, since its value projection leans on T more than the random baseline.

I'm checking whether L3H3's value projection reads the key's own token, which would fit an induction-style copy, and it does lean on that token more than the random baseline. Its keys don't pull in the positional code, so this looks like induction rather than something a shuffled-text model would need. I'm also wondering whether earlier lines, whose character bags are shared, could matter for the current line, though that feels like a stretch.

So L3H3 isn't restricted to the current segment and attends broadly by content, which fits an induction-style copy. Meanwhile the layer-1 and layer-2 heads seem to attend within the segment while steering away from matching tokens, and I'm wondering how that works mechanistically without good embedding magnitudes to check against.

I'm estimating the size of L0H5's output by taking the Fin-component of the normalized input and multiplying by its singular value of about 16. If that component is around 3 to 5, the output could be roughly 50 to 80, which would swamp the residual stream and leave layer 1 mostly seeing the Fout feature. That seems odd, though it might still be possible.

I'm wondering if L0H5 acts as an attention sink, with its large output on early positions creating a massive activation that later heads key on regardless of query.

I'm thinking the sink idea through: if L0H5 averages a position-0 feature over the prefix, its output would decay with position, so later heads might key on early positions as a default target.

I'm wondering whether the Fin direction is just a position-0 indicator, which would make L0H5's near-uniform averaging produce an output that decays with position. I'm also checking whether Fin overlaps with the positional subspace P the way it seems to, since that would tie the sink idea to the positional code.

I'm weighing whether the as-is ordering holds up: L0H3 acts like a positional copy head, and L3H3 reads its output for an induction-style match that copies the key's own token. The layer-1 and layer-2 heads, which stay within the segment and avoid matching tokens, don't fit that picture as neatly, so I'm still checking how well the two readings line up.

I'm weighing whether layer-1 heads avoiding their own character could be a reasonable way to gather information about other characters, since the query position always scores highly against itself and the head needs to look elsewhere.

I'm comparing L0H3's offset against L0H0 to see whether it's a true previous-token head, and checking whether L3H3's previous-previous match with the key's previous character would fit an exact offset of one.

I'm considering whether L0H3 is a strictly previous-token head or a broader local one, since the prev-prev matching in L3H3 would fit an offset-2 pattern. I'm also thinking about whether the positional subspace follows a rotation per step, which could let me infer the offsets directly from the attention structure.

I'm considering whether L0H0's score could be a monotone function of position, so that a simple eigen-angle reading would show whether the offset preference is broad or sharply local.

I'm weighing whether L0H5's output could behave like a position-0 sink, since it averages a feature that might mark only the first position. I'm also checking if the 1-3 step offset in L0H3 is a strictly previous-token head or a broader local window, and whether that would explain the prev-prev matching in L3H3.

I'm now going through the six setups to see which fits the head patterns best. As-is Python should show strong structure with indentation-tracking and identifier induction heads, while as-is English and romaji should show prev-token heads and moderate induction. For the shuffled versions, the heads should instead be broad bag-of-characters heads, with Python's newline and line-length patterns being the main exception.

Now I'm turning to L3's other heads and whether any of them are near-uniform averagers. I'm also wondering whether L0H3's OV, which copies the full token identity rather than just newline-ness, would matter for a shuffled line's start.

I'm now looking at whether L0H3's shifted reading gives L1 a running "word so far" view of the as-is text, with each position pairing its character with the one before it. In shuffled text that same pairing would just act as a shifted bag, so the heads would have less reason to prefer it.

I'm weighing the as-is versus shuffled cases: the sharp relative-position head and the induction-like L3H3 both point toward as-is text, since segment heads there can combine current and previous tokens into a word-prefix view. I'm still checking whether the sharpness of L0H3's query-key structure really holds across its many singular directions.

I'm checking whether the segment boundaries are newlines or spaces: spaces would mean the as-is text is split into words, while newlines would mean lines. I'm also trying to identify individual token directions in L0H5's key, perhaps through sparse coding of the first-layer weights, though that looks hard to do directly. Then I want to see whether the massive-activation view explains the structure.

Checking the scale of L0H5's output: with Fin components of a few units, the 16-scale singular value pushes its output to roughly 50 to 160, which looks massive next to typical residual norms. I'm now wondering whether the layer-1 norm will flatten that out.

I'm checking the layer-1 LayerNorm weights on the output direction, since a massive component there would squash everything else. I'm also looking at whether the values ignore that direction while the keys lean on it, which would make the sink a place for heads to attend without contributing anything.

I'm wondering whether L0H5's Fout is a sink mechanism rather than a segment detector: position 0 would carry a massive Fin component, and later positions would receive a decaying share of it through averaging. But the head isn't restricted from attending to position 0, so the decay wouldn't be clean.

I'm thinking about whether the sink might just be newline or space tokens acting as no-ops, which would make the sink-versus-segment question matter less. Rather than settling that directly, I want a different way to tell the corpora apart, one that doesn't depend on naming individual tokens.

I'm checking whether the tokens l, q, v, and x ever appear in the romaji text, since loanwords or embedded English could bring in a few. If they barely occur, their embeddings would sit near their random initialization, and the model would have no reason to learn to read them.

I'm checking how many distinct tokens each corpus actually uses, since romaji leaves several letters almost unused, which would make its effective token rank around 24 versus about 28 for English and Python. I'm also looking at the spectrum of L0H3's OV map, which drops off smoothly after about 18 strong values, and wondering whether rarely occurring letters like j, q, x, and z still get enough training signal to shape that tail.

I'm checking whether vowel-versus-consonant structure in romaji shows up in the layer-0 MLP neurons, since a vowel-detector neuron would reveal which letters the model groups together.

I'm weighing whether the token embeddings could be recovered from the structure of the weights, though that looks hard without the embeddings themselves. The output format expects probabilities, and the evidence may be weak, so I'm thinking about which signals would separate the corpora most reliably.

I'm wondering whether Python's copying shows up in the heads: its repeated lines and boilerplate should make induction stronger than in English or romaji. Shuffling would still break any sequence to copy, though line-level bags might carry some signal. Next I want to look at the QK and OV structure of L3H3 and L3H4 to see how induction-like they really are.

I'm checking whether the query side also reads the Fout component, since queries seem to pick it up at two to three times the baseline. To see if that matters, I want to compute the Fout-to-Fout interaction on its own, which would show whether matching keys share a segment or are drawn to a shared sink.

Now I'm computing the Fout-by-Fout block of the attention matrix for the layer-1 heads, and I'll do the same for the layer-0 heads to compare.

