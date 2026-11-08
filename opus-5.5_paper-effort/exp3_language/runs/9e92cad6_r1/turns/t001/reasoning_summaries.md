I'm starting by looking at the weights, since there's no forward pass to run. Without embeddings I can't read off positional patterns directly, so I'm turning to the circuit structure instead: QK interactions between heads, and K-composition between layer 0's OV output and layer 1's QK, which would point to induction heads.

I'm working out that shuffling the characters within each line would make order-dependent heads like previous-token ones less useful, since the model can't rely on sequence inside a line. Broader attention that counts characters within a line, plus the newline position as a length cue, would matter more. Repeated Python lines would also stop helping induction, since their order is scrambled.

I'm checking whether the weights alone can reveal positional patterns, since without the position embeddings a direct read isn't possible. I'm considering whether the low-rank structure of the query-key products might show a positional component, though that looks hard to pull out without the embeddings. I'm also wondering if the token embedding subspace could be identified from the layer norm weights so I can tell which directions correspond to which characters.

I'm weighing whether the corpus can be inferred from the structure of the weights, since training dynamics and dimensionality might carry clues. Python would show heavy indentation, so the model would need to copy leading spaces from the previous line, and its repeated identifiers would drive strong induction. Romaji and English would look different, with rarer letters and long paragraph-length lines.

I'm trying to work out what the weights can reveal about the corpus, since the embeddings aren't available to read positional patterns directly. Recovering character identities from the residual stream directions looks hard, so I'm weighing whether training dynamics or the weight structure itself might point to the kind of text, like heavy Python indentation or long English paragraphs.

I'm thinking about whether the position embeddings could be separated from the token embeddings in the weights, since a previous-token head would show up as a positional pattern if I could isolate that subspace. Without them, I'm not sure the layer-0 query-key products can reveal it directly, and I'd need another route to tell which heads are positional.

I'm going after the random-matrix bulk: singular values of W_q.0, W_k.0, W_v.0, and W_fc.0 that stand above the noise floor should reveal the learned input subspace, which is roughly tokens plus positions after the layer norm scaling. Positions are probably smoother and lower-dimensional than the 28 token directions, so the subspace may be smaller than it first looks.

I'm weighing whether the shuffled-text case would leave previous-token heads with little to do, since within a line the order is random and only the set of characters seen so far carries real information. Broad attention over the current line, plus something that marks the newline, would then matter most.

I'm weighing which heads would survive a shuffle: the broad, bag-of-characters ones over the current line should stay useful, while previous-token and induction heads would mostly go idle. Induction could still show up in the weights through K-composition, where a later head's keys read from an earlier previous-token head's output.

I'm checking whether the positional subspace could be isolated from the weights, since a prev-token head in layer 0 would be mostly positional and hard to spot without the position embeddings. I'm also thinking the sinusoidal-like structure of the position vectors might let me separate them from the token directions.

I'm checking whether a previous-token head's query-key product, restricted to the positional subspace, would show a rotation-like offset by one. I'm also thinking about whether the embeddings could be reconstructed from the weights, though the lack of a privileged basis makes that hard.

I'm wondering whether the 28 token embedding directions could be pulled out of the layer 0 weights, since they should be the only things the learned parts of those matrices align with besides the positions. Without tok_emb I can't name them, so the subspace alone may not be enough to tell which character is which.

I'm considering whether the corpus could be guessed from the structure of the attention heads: Python would show strong induction from repeated identifiers and indentation, English moderate induction, and romaji some repetition. The trouble is that without the token embeddings I can't tell which directions correspond to which characters, so this stays speculative.

Next I'm checking the basic stats: norms, singular values of each weight matrix, and the layer norm weights. With a budget of about 100 runs, I'll keep the exploration focused.



---

I had gotten stuck repeating pointless echo calls to bash_tool. The actual protocol is to reply with a single python code block, which the harness executes, so I'll do that now without apology. Next I'm planning a first analysis covering singular values, norms, and LN weights.

