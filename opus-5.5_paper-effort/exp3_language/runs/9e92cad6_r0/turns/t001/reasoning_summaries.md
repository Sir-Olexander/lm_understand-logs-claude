I'm starting by looking at the weights directly, since there are no embeddings to map back to tokens. My plan is to inspect the attention heads through their structure: the query-key and value-output products, and how heads compose across layers, especially K-composition as a route to induction behavior. I want to compare shuffled text against ordered text, because without order inside the lines the model would have to rely on broad, bag-of-characters attention within each line.

I'm weighing whether the corpus is Python, romaji, or English, which is hard to judge from the weights alone. Python's indentation and short, repetitive lines would favor induction heads, while romaji's small alphabet might show up in how the embedding subspace is used, though rarely used tokens stay near their initial values, making that tricky to read.

I'm now estimating which heads might be induction-like by looking at how earlier heads feed into later ones, since without embeddings I can't check copying against the unembedding directly. I'll compare the composed QK-OV products against a baseline to see whether any pair stands out.

I'm now thinking about how shuffling would affect the attention patterns: without intact line order, a head that copies from earlier characters in the line would behave differently, and it's unclear whether the copying signal would be positive or negative. The distinction between shuffled and ordered text seems to hinge on whether spacing and indentation carry predictable structure.

I'm now working out how the shuffled text would change things: within a line the characters have no fixed order, so previous-token heads wouldn't help, and the heads would spread attention evenly across the line and perhaps the neighboring lines. The open question is whether any heads still show sharp positional attention to the previous token.

I'm checking whether layer 0's positional attention can be read from the weights alone, since the positional embeddings aren't available. I'm looking at the singular values of W_q and W_k for each head, to see whether a previous-token pattern would show up as a rotation-like structure in a positional subspace.

I'm checking whether the antisymmetric part of the positional QK bilinear form is large, since a rotation-like structure would point to a previous-token head, while a symmetric form would suggest something that attends to itself or broadly.

I'm checking whether layer 0 can carry sharp previous-token attention from the positional subspace alone, and whether token content could steer heads toward newlines so they gather only the current line's characters.

I'm checking whether the positional subspace can be identified from the layer-0 weights, since the learned parts of W_q, W_k, W_v and W_fc should line up with the span of the input embeddings. That would show which directions in the residual stream carry position versus token identity.

I'm working out whether the token embeddings can be separated from the positional ones in the layer-0 weights. Tokens that never appear in the corpus would get no gradient, so their embedding rows would stay near initialization, which could help tell the two apart.

Now I'm checking whether the Hepburn letters that never appear in the corpus (l, q, v, x) keep their random initial embeddings, since weight decay over training should barely shrink them.

I'm wondering whether the token subspace can be picked out of the layer-0 weights, since the learned parts of those matrices should lie in the span of the inputs. Romaji has fewer distinct tokens than the other candidates, but a gap of a few dimensions would be hard to detect. Line length and indentation might be easier cues: Python has short lines with leading spaces, while English and romaji text have long paragraph-like lines, so I'm checking whether the weights reveal any of that.

I'm wondering whether the embeddings can be partly reconstructed from the layer-0 weights, since their learned parts should lie in the span of the inputs. Python, English, and romaji would each leave a different signature in that subspace, though the differences look subtle. Still, repetition patterns like identifiers or names might show up in which heads attend broadly and which stay local.

I'm thinking about whether the learned parts of the layer-0 weight matrices lie in the span of the inputs, so the top singular directions could reveal the token and positional subspaces. Random init noise sets a baseline, and the learned components should stand out above it. Some heads might then be recognizable from attention to line starts or previous-line indentation, which would point toward Python.

I'm weighing whether the corpus could be guessed from the output-side weights, since the last layer's projection gradients should live in the span of the output unembedding. The effective alphabet size and letter frequencies differ between Python and English, but without the embeddings I can't read them off directly.

I'm now looking at whether the output-side weights can reveal the corpus: the learned columns of the last projection should lie roughly in the span of the output vocabulary. Tokens that never show up as targets still get a downward push from the softmax gradient, so every letter stays represented. The question is whether the effective number of distinguishable outputs, around 26 for romaji versus 28 for the other candidates, is large enough to show up in the singular value spectrum.

I'm checking whether the output-side weight gradients of the last projection lie mostly in the span of the output unembedding, which would make the effective output dimension, around 26 or 28, show up in the singular value spectrum of that matrix.

I'm now checking the output-side spectra, W_proj.3 and W_o.3, to estimate the effective output rank. A clear drop near 26 to 28 might show how many output letters the corpus actually uses, though Adam's elementwise updates may blur that structure.

I'm noting that the unused tokens' input embeddings get no gradient, so the layer-0 weights should span a smaller input subspace. Positional dimensions still take up much of that space, which makes the difference hard to see. Next I'll compute the shapes, norms, singular value spectra, and layer-norm statistics for the main matrices.



---

I'm going to start over, since the last attempt got cut off. Before anything else I want to run code to analyze the weights. The reply has to be a single code block with nothing around it, so the first thing I write will be code. My plan is to compute weight norms for each layer and each head, and to note that without the embeddings I can't tell positional from token components directly, so I'll look at the structure of the residual stream subspaces instead.

I'm weighing which heads could still do useful work on shuffled lines, since order inside a line is random and lines are shuffled independently. Broad attention back to the last newline seems more plausible than previous-token or induction behavior, which would have nothing to latch onto.

Still working through how to tell the token and position components apart, since the embeddings aren't available to read directly.

I'm checking whether the attention structure can reveal positional behavior without the position embeddings, and whether the bilinear form W_q·W_kᵀ for layer 0 looks asymmetric in a way that would suggest a previous-token head.

I'm checking whether the W_q·W_kᵀ structure for layer 0 shows rotation-like behavior that would point to a positional, previous-token head, since the embeddings aren't available to confirm it directly. Broad heads that attend across the line look more plausible for shuffled input, and I'm also thinking about whether later layers could show K-composition with that earlier head.

Without the embeddings, I'm thinking the token-embedding directions might show up indirectly, since the directions that carry real input signal should have larger weight norms in the layer 0 projections than the unused directions, which only decay under weight decay.

Working out how the input subspace reaches layer 0's projections: the residual stream there lives in the span of token and position embeddings, about 156 dimensions, and layer-norm's per-dimension scaling carries it into W_q.0's input space.

I'm comparing the input subspace each weight matrix reads from: W_q.0, W_k.0, W_v.0, and W_fc.0 should all share the trained directions, while their random-init components are independent across matrices. The shared subspace should be roughly 156 dimensions, and I'm also checking whether the layer-norm weights ln1_w.0 and ln2_w.0 reshape it.

I'm now checking whether Adam's elementwise normalization undermines the idea that the trained weights stay confined to the input subspace. It probably does, so the trained structure will mostly sit inside that subspace while the optimizer noise spreads outside it.

I'm thinking about whether the layer-0 heads could attend within the current line by keying on the most recent newline, and how that would show up without the embeddings. Telling corpora apart is the harder part, so I'm looking at cues like indentation, where leading spaces stay in the text.

Now I'm comparing how the different corpora might look in the weights: Python has lots of indentation and repeated identifiers, romaji has regular vowel-consonant rhythm, and English has long paragraph lines. I'm wondering how to get at the token structure when the embeddings aren't available to read directly.

Now I'm weighing whether the token embedding structure can be recovered from the layer-0 weights, since the embedding matrix itself isn't available. I'm considering whether the final output projection, which reads from the last MLP through the layer norm into the 28 logits, could reveal the token space instead.

I'm checking whether late-layer writes, like W_proj.3 and W_o.3, are confined to the roughly 28-dimensional subspace that actually feeds the output logits. Writes in other directions shouldn't affect the loss apart from scale changes from layer norm, so the gradient should live mostly in that subspace, plus a component along the residual itself.

Now I'm thinking about whether the output-side directions can tell me anything, since the last layer's writes should mostly sit in a roughly 28-dimensional subspace that feeds the logits. The 28 token directions there aren't identifiable without the unembedding, so I'm looking at whether broader statistics of the weights could separate the corpora instead.

I'm looking at whether the layer-0 QK pairing might reveal token matching, so I'm considering the induction-style composition with the layer-1 query and key, since the token embedding itself isn't available to check directly.

I'm checking whether the layer-1 induction-style QK product, with the layer-0 previous-token OV circuit folded in, behaves like a token-matching operator. If so, its diagonal entries over the token directions should be strongly positive when the query and key tokens agree.

Checking whether the positive-eigenvalue fraction of the layer-1 QK-times-OV product shows the self-matching pattern expected of an induction-style head, since the embedding matrix isn't available to test it directly.

I'm working out how a previous-token head would look in the weights: if the position embedding is roughly sinusoidal, the bilinear form between query and key positions should act like a rotation by each frequency's angle, so its eigenvalues would come out complex with phases near the unit circle.

I'm checking whether a previous-token head would show up as a rotation-like form on the position subspace, with eigenvalues near the unit circle. The symmetric and antisymmetric parts of the query-key product would tell me whether the head sharply picks out the position just before each token.

I'm looking at whether a previous-token head would show up in the query-key product. The antisymmetric part should carry the offset-by-one signal, since low-frequency rotations are mostly symmetric while higher frequencies contribute more antisymmetric weight. I'm also noting that the LN bias terms add constant vectors to queries and keys, which shouldn't affect the softmax across keys, so I can set them aside.

I'm looking at whether the layer-0 weights could be read through the centered token and position embeddings instead, since those inputs get normalized per position. The trained structure of W_q.0, W_k.0, W_v.0 and W_fc.0 should then sit inside that roughly 157-dimensional input subspace, scaled by the layer-norm weights.

I'm weighing whether the induction-style matching and the previous-token behavior can be read from the weights at all, since the embeddings are missing. A simpler route might be to look at the shuffled-versus-unshuffled difference: if the lines are shuffled, those order-dependent heads have no use, so they should look broad instead.

I'm weighing whether the shuffled and as-is models would differ in their attention. With shuffling, the heads would have no order to exploit, so they should spread out over the characters in the line and key on newlines, while the as-is model should show induction-style composition and rotation-like previous-token heads in layer 0. The open question is how to recover the positional structure without the position embeddings.

I'm checking whether the induction-style composition can be detected without the positional embeddings, by chaining the layer-0 value-output circuit into the layer-1 query-key product. If that composed operator is roughly symmetric and dominated by positive eigenvalues, it would point to token matching.

I'm considering whether the positive trace of the composed layer-0 and layer-1 operator, normalized by its Frobenius norm, could separate induction-style matching from a random baseline, since a positive trace would suggest the heads match tokens.

I'm weighing whether the shuffled data would leave the heads with no order to exploit, so they'd spread out and mostly count characters. The next question is whether the same-token attention pattern could separate the shuffled model from the as-is one, since the line's letter makeup is the main thing a bag-of-characters predictor can use.

Still weighing how the shuffled and as-is models might differ: shuffled lines would leave heads broadly attending over the current line and the previous line's indentation, while the as-is model would carry more order-dependent structure.

I'm thinking about whether a bigram-style structure in the layer-0 token subspace could distinguish romaji from English, since romaji has a strong consonant-vowel alternation. Line length matters too: longer paragraphs would leave a 128-character window mostly inside one line, so the heads would look broad.

Romaji text barely uses l, q, v, or x, so only about 22 tokens would have trained embeddings. I'm asking whether the effective rank of the token embedding subspace, read from the weights, shows that narrow use.

I'm trying to see whether the trained part of the layer-0 input space can be pulled apart into token directions and position directions, even though the embeddings themselves aren't available. One idea is to look at the layer-0 MLP neurons for token-specific responses, though that's hard to read without the embeddings.

I'm considering whether global weight statistics could separate the two training conditions. Shuffled models might show near-uniform layer-0 attention with small query-key norms, since bag-of-characters prediction needs no sharp position selection, while an as-is model would need large position-dependent scores for previous-token behavior.

I'm looking at the output side now: if the corpus is romaji, only about 22 letters plus space and newline ever get predicted, so the unused letters would just need strongly negative logits, which the output bias could handle. That would make the effective output subspace smaller than the full 27 dimensions, whereas Python and English would use nearly all of them.

I'm checking whether the output-side subspace is smaller than the full vocabulary: if the corpus is romaji, the unused letters would sit far down in the logits, so the column space of W_proj.3 and W_o.3 might show roughly one fewer dimension per unused token. The input side should show the same thing for the token embeddings, though the untrained rows for unused tokens won't be learned directions anyway.

I'm checking whether the last-layer writes need to vary across the roughly 24 tokens that actually show up in romaji text, since the unused letters would just get pushed down by the output bias. That would leave about a 23-dimensional varying subspace in the logits, which I could compare against the full 27 dimensions to see if the column space is smaller.

I'm going to check the singular value spectra of the final layer's output projections to see if there's a knee, since that would show how many dimensions the output actually uses. The layer-3 writes should line up with the unembedding subspace, though earlier layers also feed the residual stream that gets read at the end.

I'm checking whether the trained part of the final layer's output projection lines up with the output subspace, since the dominant singular directions of W_proj.3 should reflect how many tokens actually need their logits to vary. Tokens that never appear as targets, like the unused romaji letters, should add at most one extra direction, so the effective output dimension would be a bit smaller than the full vocabulary. I'm going to look at the singular value spectrum of that projection for a clear knee.

Checking whether the spectrum of the layer-3 output projection shows a clear knee, since that would reveal how many output dimensions the model actually uses. I'm also thinking about the same comparison for the value and output weights of layer-0 attention heads, plus how composition scores between heads could point to previous-token and induction behavior.

Next I'm checking the basic statistics: the Frobenius norms of each head's QK and OV matrices, the layer-norm weight distributions, and the singular value spectra of the final layer's output projections.

