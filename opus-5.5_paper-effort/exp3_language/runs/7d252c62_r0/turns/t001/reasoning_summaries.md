I'm starting by looking at the model's weights, since there are no embeddings to read from directly. The main question is how shuffling lines would change things: within a line the order wouldn't matter, so attention should spread across the characters in that line, without previous-token or induction heads, with position only mattering at line boundaries.

I'm working out how the shuffling would affect things: order within a line stops mattering, so attention would spread across the line's characters, and only line boundaries carry positional signal. Python's repeated identifiers across consecutive lines might still give induction-like patterns, while romaji's consonant-vowel structure would lean heavily on previous-token attention.

I'm weighing whether the token embedding subspace can be recovered from the weights alone, since the positional part is unknown. The first layer's layer norm and the query-key matrices might reveal which dimensions carry token information, but that's hard to pin down without more structure.

I'm looking at how to identify the token subspace from the weights, and comparing composition scores between layer 0 and layer 1 heads, since high K-composition from a previous-token head into a later head would point to induction behavior.

I'm checking whether the positive eigenvalues of each head's W_O W_V, taken in the full residual space, mean the head copies the features it reads, which would fit an induction-style copying role. I'm also wondering whether the positional subspace could be found from the weights, since a layer-0 previous-token head should attend to position i-1 there, but I don't know the positional embeddings yet.

I'm checking whether the antisymmetric-to-symmetric ratio of W_q^T W_k in these layer-0 heads separates positional rotation-like heads from content-matching ones.

I'm weighing whether the shuffled data would make the model rely on a bag-of-characters estimate within each line, since the order of characters no longer carries information. Given that, previous-character heads would matter less, while attention spread across the current line's characters and the line-length signal for predicting newlines would matter more.

I'm now thinking that if the lines are shuffled, the characters within a line become interchangeable, so the previous character tells me little beyond what any other character in the line does. Consecutive lines might still share identifiers in Python, and romaji or English paragraphs are long enough that the context amounts to a bag of characters, so the heads would spread attention broadly across the line and look for the newline.

I'm checking whether attention heads that spread evenly over the context can be spotted from small query-key norms, though that averages over the whole context rather than just the current line. Restricting to the line would need something like heads that attend to the most recent newline, which an early layer with recency bias could find, and later layers could then compare against that.

I'm planning to compute basic norms for each head's W_q, W_k, W_v, and W_o, along with the spectral properties of the QK matrices. I also want to see whether the token-embedding subspace can be inferred from the layer-0 input, and I'm wondering whether the number of effective token directions could tell romaji apart from other text, since it uses a limited set of letters.

I'm checking whether the romaji letters that never show up, like l, q, v, and x, would have barely-trained embeddings that stay near their random initialization. I'm not sure the weights alone can tell me that, since I don't have the embeddings to inspect.

I'm weighing how the shuffled-versus-original distinction changes what the model needs to do. Once the order inside a line stops mattering, the model can't rely on position to predict the next character, so it would have to attend across everything since the last newline and treat that stretch as an unordered bag.

I'm weighing whether the layer-0 heads that read the previous token would form a K-composition chain with later heads. I'll check that by scoring each layer-0 head's OV against later QK matrices and comparing those scores to a random baseline.

I'm working out whether the positional subspace P can be read off the first-layer query and key matrices, since the rows of W_q.0 only see the input space spanned by token and position embeddings, and the gradient reaches them through the layer-normalized input.

I'm working out the decay factor for the random initial weights in W_q.0: with decoupled weight decay of 0.01 and a learning rate around 1e-3 over roughly 30k steps, the shrinkage comes to about exp(-0.15), which is small enough that the orthogonal components would still be visible.

I'm looking at the singular spectrum of the first-layer input weights: about 158 directions should carry the trained signal, while the other 226 should sit at the random-noise level, with the bulk edges set by the matrix's dimensions.

I'm checking whether the input subspace is really rank 156, since adding the layer-norm rescaling per token and position could widen the span beyond the token and position sums.

I'm figuring out how the 158-dimensional input subspace splits into token and positional parts, since that split would let me pick out the positional directions in the weights.

I'm now checking whether the 28-dimensional token subspace can be pulled out of the layer-0 input, even though the embeddings themselves can't be recovered exactly. I'm also wondering whether the spectrum of the token-side bilinear form means anything, since without orthonormal embeddings it may not be basis-invariant.

I'm now asking whether the corpus could be told apart from the weights alone. Python would show up as short lines with leading indentation and lots of repetition, English prose as long paragraphs with rare newlines, and romaji as short dialogue lines with a strong consonant-vowel pattern. Pinning that down from the weights looks hard, though, and I'm considering whether dead-neuron counts or the sparsity of the MLP might carry the signal.

I'm considering whether the token set could be identified from the spectrum of the layer-0 matrices restricted to that input subspace, since the number of characters used clusters the token directions. Romaji would use only about two dozen letters, with l, q, v, and x nearly absent, while English and Python draw on all 26, so the count of active directions might separate them.

I'm noting that the romaji letters rarely used, like l, q, v, and x, would get little gradient, so their embeddings would stay close to random initialization, and the weights would only show small trained components along those directions.

I'm now considering whether the rare romaji letters, like l, q, v, and x, could show up as weaker trained directions in the layer-0 weights, since they barely receive gradient. The catch is that the token and positional directions are mixed together in that subspace, which makes counting the active token directions difficult.

I'm counting how many singular values of the stacked layer-0 input weights rise above the random-noise bulk, expecting roughly the frequent-token directions plus a few positional ones to stand out.

I'm wondering whether the positional directions can be told apart from the token ones in that subspace, since the 127-odd position vectors may have far less effective rank than the token set. I'm also looking at whether layer 1's residual stream, which mixes the embeddings with the outputs of attention and MLP layer 0, would show the same structure.

I'm checking whether sharp offset structure in the positional part of the query-key product would show up as a previous-token head, since that would mean the model depends on exact position, which shuffling would destroy.

I'm looking at how the high-frequency rotation terms make the antisymmetric part of the positional QK matrix matter, which would separate sharp previous-token attention from heads that just attend to themselves. I'm also planning to check the overall structure with an eigenvalue analysis of W_q^T W_k, which has rank 64.

I'm checking whether the layer-0 previous-token and layer-1 induction heads compose: the layer-1 query-key matrix applied after the layer-0 output-value matrix should map a token onto itself, which would show up as positive real eigenvalues in the token subspace.

I'm working out how the induction circuit composes across layers: the layer-1 key reads the layer-0 output-value map applied to the previous token, while the query reads the current token directly, so the combined matrix should map each token onto itself.

I'm checking whether the induction circuit shows up as positive eigenvalues in the composed matrix B, which maps the current token onto the previous token's contribution through the layer-1 query-key and layer-0 output-value paths, since that doesn't depend on knowing the token basis.

I'm weighing whether the eigenvalues of B could show the induction structure even without knowing the token basis, since positive real eigenvalues are basis-invariant. I'm going to compare the trace of B with the sum of the absolute eigenvalues to see how strongly the symmetric, matching part dominates.

I'm checking whether the induction circuit is visible by computing, for each pair of an earlier layer-0 head and a later head, the product of query, key, and value-output matrices, then measuring how much of its eigenvalue mass is positive.

I'm checking whether the layer-0 head is truly previous-token by looking for rotation structure in its positional QK terms. The matching test is the positive-eigenvalue share of the composed layer-0 OV and layer-1 QK product, since positive eigenvalues mean the query reads the same feature the earlier head wrote from. For the corpus question, I'm also considering whether indentation patterns, with runs of leading spaces, could show up in the attention heads instead of the Wikipedia-style long-paragraph structure.

I'm weighing whether the romaji letters that barely appear, like l, q, v, and x, would leave a detectable mark in the embedding directions, compared with English and Python, which use all 26 letters heavily enough to train every direction.

I'm considering whether the rare romaji letters leave any detectable trace, and how the token and positional directions might be separated in the layer-0 subspace, perhaps by using the OV matrices of the layer-0 heads, since a previous-token head's output should read token identity rather than position.

I'm reasoning about the layer-0 weight space: the input subspace of the value and output weights should be mostly token directions, since only about 28 distinct tokens exist. I'm checking how many singular values of each head's value matrix stay large, since the trained token part should have rank at most 28 out of 64.

I'm counting how many singular directions in the stacked layer-0 input weights carry real signal, since the trained part should sit in the span of the token and positional embeddings. Then I'm looking for a way to separate the token directions from the positional ones.

I'm looking at the gradient with respect to the residual stream, which for the final layer runs through the unembedding and the final layer norm, so its span is mostly the 28 output directions plus some layer-norm projection effects.

I'm now looking at the output side of the last MLP layer, where W_proj.3's trained columns should sit in roughly the unembedding subspace, about 28 dimensions. I expect the left singular spectrum to show a few large values for the frequent tokens and noise for the rare romaji letters, whose output rows probably stayed near their random initialization.

The gradient for rare output tokens is tiny, so the trained output space of W_proj.3 should have rank roughly equal to the number of frequently predicted tokens minus one, since the softmax gradient sums to zero. For romaji that comes to about 23 significant directions out of 24 active tokens.

I'm counting the Hepburn letters that actually appear in romaji: vowels a, i, u, e, o plus consonants like k, s, t, and the others, leaving out l, q, v, and x, which gives about 22 letters. Adding space and newline puts the active output set near 24 tokens, so I'd expect roughly 23 to 24 significant directions, though that count depends on how dominant the frequent tokens are.

I'm now weighing how rare letters like 'q', 'z', 'x', and 'j' in English still get enough occurrences in a billion characters to train their directions fully, even though each is well under a percent of the text. In romaji, 'l', 'q', and 'x' should be almost absent, with only a few stray Latin-script words from the source texts possibly giving them a trace, so their directions would stay weakly trained.

So the rank of W_proj.3's trained output directions should track the number of tokens actually predicted, and the final layer norm adds a few extra directions along the normalized residual.

I'm checking whether the x̂ projection terms, which point along the normalized residual, would blur the count of trained output directions. Their size relative to the unembedding part depends on how much the gradient aligns with x̂, and summed over many positions that could spread the spectrum enough to hide the vocabulary-size signal.

I'm checking whether Adam's per-coordinate normalization distorts the low-rank output subspace of W_proj.3, since its trained update is no longer a clean sum of gradient outer products.

I'm checking how Adam's per-coordinate scaling reshapes the update: if the second-moment term is roughly a rank-one outer product, the normalization is a diagonal rescaling on each side, which keeps the rank of the trained part but tilts its subspace.

I'm turning to whether Python and English can be told apart by their output. Python's heavy indentation and frequent newlines should show up as strong directions that English, with its rare line breaks and no leading spaces, wouldn't produce.

I'm considering whether a leading space survives normalization in a line like "(hello)", since the rule about turning non-letters into spaces could make the start ambiguous. Python's indentation and frequent newlines should produce strong directions that English, with rare line breaks and no leading spaces, wouldn't.

I'm weighing whether attention-head patterns could separate Python from English, since Python's indentation and frequent newlines should drive heads that attend to line starts. The unembedding rank alone won't tell them apart, since both use the same 28 tokens, so I'm looking at the input-side token subspace instead.

I'm thinking about whether the bigram structure could be recovered from the layer-0 attention and MLP paths, since the direct token-to-output route isn't available. I could also use the frequency pattern in the trained singular values, though Adam's normalization makes that weak evidence.

Since there are no direct labels, I can't map input tokens to output tokens through a copy head's OV circuit, so the only handles left are structural ones like how many tokens are active and the shape of the spectrum. I'm also wondering whether the bigram statistics could be read out of the layer-0 prev-token head and MLP0 path.

I'm weighing whether attention patterns could tell Python from English, since Python's repetitive identifiers and keywords and its heavier reliance on looking far back to earlier lines might show up in the heads. Without direct token labels, I'd have to rely on structural signals like the spectrum and attention range, and I'll check those against actual data.

I'm going to check each layer-0 head's positional QK for antisymmetry, since a prev-token head needs that component to prefer position i-1 over the current position.

I'm checking the antisymmetric part of each layer-0 head's positional QK to see which heads look for the previous token. Then I want to know whether the trained positional directions occupy only a few dimensions, which I'd compare against the 28-token subspace. Next I'll print shapes, norms, and LayerNorm weight stats, plus the singular values of the main matrices.



---

I'm checking the user's note about a cut-off reply, but there's no earlier response visible, so I'll answer from the beginning. I need to run code and reply with exactly one code block. First I'll look at basic statistics of the weights, then think about inferring things without embeddings, including the idea that characters within each line are shuffled.

Without embeddings, I'd have to infer structure from the weights alone. Since the lines are shuffled, attention within a line would track character counts, and the newline might mark line position. Induction heads should be weak there, though they could still matter in unshuffled Python with repeated identifiers. Without positional embeddings I can't read previous-token heads directly, so I'll look at the QK and OV circuits and check for K-composition between early and later heads.

Next I'm looking at how to find previous-token heads without positional embeddings, since their query-key circuit is mostly positional. The idea is to check whether the positional structure forms rotation-like patterns in the weight subspace.

I'm working out how to find previous-token heads without positional embeddings, using the skew-symmetric part of the query-key matrix. If positions lie on rotating circles, a head's query-key form should show complex eigenvalues whose angle reflects a preferred offset, so I'll compute the eigenvalues of W_q^T W_k and check the dominant ones.

I'm working out how the token embeddings might show up in the weights, since the vocabulary is only 28 characters and that should leave a small low-rank footprint. Random initialization at std 0.02 gives a noise floor of roughly 0.78 for the top singular values of a 384x384 matrix, so anything trained should stand out above it.

Since the lines are shuffled, a character's best predictor within its line is the set of characters already seen there, and the line's length matters too. Previous lines keep their original order, so any copying signal would come from those. I'm checking whether the spaces are shuffled along with everything else in each line.

Previous-token heads are less useful here, since within a shuffled line the characters are interchangeable, so the best cue is the multiset of characters already seen plus the newline marking the line start. In unshuffled text they'd be central, with induction heads appearing from layer 1 on. Without positional embeddings to read directly, I'd have to look for them in the structure of the query-key matrices instead.

I'm weighing whether the token subspace can be told apart from the positional one, since the MLP in layer 0 may mostly carry token information. Rather than settle that by reasoning alone, I'll start with the basics: norms, singular value spectra, the QK eigenvalues per head, and composition scores.

I'm weighing whether Python's indentation, with its runs of leading spaces and repeated identifiers, would show up as strong induction behavior, and whether romaji's alternating vowels and consonants or English word structure would leave different signatures in the attention patterns.

I'm also considering whether vocabulary usage could reveal the corpus, since romaji would leave letters like l, q, x, and v nearly unused. I'd need to check whether unused tokens keep their random initialization, but without the embeddings I can't read that directly. Next I'll look at the MLP neurons in layer 0, which read from the residual stream after attention.

I'm trying to recover the token embedding directions from the layer-0 weights, since the trained signal should sit in a small subspace above the random-init noise floor. The idea is that the input to the first layer's norm is token plus position, so the trained updates in the query and value matrices should span those inputs, and their top singular directions might reveal the token part.

I'm reasoning that the input span of layer 0 is roughly the token directions plus the position directions, about 28 + 128 dimensions after centering. Weight decay over training should have shrunk the random initialization only slightly, so the init noise should stay in the orthogonal complement of that span. That means I can estimate the input subspace by combining the weights of W_q, W_k, W_v, and the first MLP matrix.

I'm weighing whether the signal subspace of the first layer's weights could tell me how many distinct letters the corpus uses, since a romaji corpus would leave letters like q, x, and v nearly absent.

I'm weighing whether attention behavior would separate the corpora better than letter counts, since romaji leaves l, q, v, and x nearly unused while Python and English use far more of the alphabet.

I'm comparing how the three corpora would differ: Python's short lines and indentation, English paragraphs that often run past the 128-token window, and romaji's dense syllable structure. Line length, how often newlines show up in context, and which letters appear at all could each separate them.

I'm weighing how the newline token might show up in attention, but without its embedding I can't check that directly. Maybe the positional structure in the query-key form can stand in for it, so I'm considering reconstructing the token embeddings from the layer-0 weights.

I'm going to check whether an induction-style circuit shows up by comparing the layer-1 key inputs against layer-0 previous-token outputs, since the query would come straight from the token embedding.

I'm checking whether the induction circuit shows up by composing the layer-1 query and key with the layer-0 OV matrix, looking for a copying structure restricted to the token subspace.

Checking whether the trace of this combined matrix, normalized by its eigenvalue magnitudes, separates matching behavior from random noise.

I'm working through how to tell apart previous-token heads in layer 0 without positional embeddings to check against. One route is to look for layer-0 heads whose output feeds the key side of layer-1 heads that also show matching behavior, since that would point to a previous-token head indirectly.

I'm wondering whether a head that attends to earlier copies of the same character could be spotted directly in the query-key weights, since within a shuffled line that would just be counting repeats. Broad heads with near-zero query-key structure should show up as small norms, though telling positional decay from that would need the positional embeddings, which I still don't have.

Next I'll check the basics: shapes, per-head norms, and singular values of the QK and OV matrices, plus the LayerNorm weight statistics.

I'm working out the effective query-key form by folding the layer-norm scales into the weights, and noting that the bias terms add a query-independent key preference, which is a possible positional or broad-attention signal. Next I'll run a first overview pass to check the basic structure.

