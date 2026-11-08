I'm checking whether the kurtosis counts really track tokens. The count rises with dimension because sparse features other than token identity, like previous-token features, also show up, so it isn't a clean token count. The 28 components above 30 in the T36 subspace include other things too, so that doesn't settle it. The MLP neurons show sparse ICA components throughout the full 60-dimensional space, so the test doesn't isolate tokens.

I'm considering a structural explanation for the corpus: Python code has indentation runs, frequent newlines, and repetitive keywords. A circuit could copy indentation from the previous line by matching columns since the last newline, and the shared rotation in the L1/L2 heads might be doing exactly that column comparison, which is worth checking.

I'm working out whether the b-out rotation could be a column-matching signal, and the open question is what b-out actually encodes, which I can't settle without the embeddings. The L0 heads that produce it have query-side positional structure but key-side mixes position and token, so they're only partly token-selective. Rough magnitudes suggest that selection could still be moderately strong.

I'm weighing whether the broad heads could be content-selective, and I'm leaning toward a rough prior: the final MLP output has about 27 token-like dimensions and a single copy head writes into roughly 28, which suggests many tokens are actively predicted. That argues against romaji, which lacks several letters, unless there are loanwords. The weak dimension count of 5 tilts slightly toward English over Python, and so does the single copy head.

I'm checking whether attention heads in the early layers respond to specific token pairs, like space-after-space, which would only appear in Python indentation. I'm also wondering whether a positional offset of about four spaces shows up in the L0 heads, and whether later heads encode that offset with rotation, though those components look very small so far.

I'm looking at L0H3's positional attention: its dominant positive eigenvalue means it mostly matches positions along one direction, so it probably attends to nearby or similarly placed positions rather than doing anything content-specific. I'll label it as a local head for now and then check how large its output is compared to the other L0 heads.

I'm weighing whether the L0 heads' key-side preference for certain tokens could identify newlines, since they'd be the line separators that stand out in a corpus. I'm considering checking the value vectors of the broad heads against the token direction to see whether they align with one token.

I'm weighing whether the broad heads prefer different tokens, since their key-side preferences look mixed rather than pointing at one token like newline. Then I'm asking what the designers would treat as decisive for the corpus: a smaller alphabet would show up as rank deficits in token-reading weights, but the roughly 27 token-like output dimensions argue against romaji. Python versus English is still open, with indentation structure pulling toward Python while the modest copy-head behavior pulls back.

I'm weighing the corpus question with rough priors: English around 0.55, Python about 0.30, romaji about 0.15. I'd expect a sharp newline-detecting head in Python, since indentation depends on line position, but none of the L0 heads I've checked is that sharp, so that points away from Python.

I'm weighing whether the L1 heads could be doing the work instead: they have a strong query-independent preference for certain key tokens, plus recency from their bias and positional terms, so they might be attending to the last space or newline. That would fit English word starts, which would be useful for predicting the next letters, but I'd still need to check how concentrated that preference is on a single token.

I'm wondering whether the L1 heads all favor the same token direction, which would point to the space token as a shared delimiter. Earlier their output carries token identity, so they seem to attend to varied prior tokens rather than only spaces. I'll check how their value reads compare, keeping in mind the analysis budget.

I'm considering whether the L1 heads' token-to-token preferences could reveal which earlier tokens matter given the current one, though that reflects usefulness rather than frequency. I'm also wondering whether the MLP0 token-detector neurons might map onto individual token embeddings, so that the number of neurons loading on each component tracks how common a token is, but that still wouldn't tell me which token it is.

I'm thinking about whether the output token directions from the final projection can be matched to input token directions from the first MLP's sparse components, using the copy head's mapping from input token to output token. Matched pairs would give me a set of unlabeled token identities, and I could then wonder whether later linear paths could approximate a bigram structure.

I'm comparing what English and Python would each look like at the character level. In English, a space never repeats and never starts a line, while in Python spaces run together for indentation. So a model reading English could just predict "not a space" after a space, but in Python it would need to know whether everything since the last newline has been spaces.

I'm considering whether the L1 heads' query-independent token preferences could reveal a delimiter, though that would still leave the identity unlabeled. Without token labels, the newline-versus-space question stays hard to settle directly.

I'm checking the value matrix's singular values for the previous-token head against what random noise would produce, and they sit close to that noise level, so the learned part isn't clearly separated yet. I'll also look at the scale of the output-value product, which is small enough that it could be mostly noise.

I'm trying to separate the output projection's directions from the positional ones, since the positional reading may take up some of the roughly 35 to 40 meaningful dimensions. I'm not sure the 24 positional directions I have capture all of that, so I'll check whether removing them leaves a clean token subspace. I'm also looking at the copy head's output singular values to see how strongly it writes token identity.

I'm checking the copy head's output spectrum: it stays significant through the 28th dimension and only drops at the 29th. If the corpus were romaji with about 24 distinct tokens, I'd expect a drop near 24 or 25, so the 28 significant dimensions point to at least 26 token types being copied, which makes romaji unlikely unless a few letters are rare.

I'm checking the singular values of the token-dimension copy block against the full 40-dimensional space, since they fall off gradually from about 4 down to 0.24 rather than showing a clean cutoff.

I'm comparing the copy-head matrix against a random-input baseline to see whether its weak tail, the 28th and 29th singular values, is real structure or just noise.

I'm sanity-checking the copy head's singular values against a random baseline, since a gradual decline from about 4 to 0.24 could just be noise. I'm also considering whether romaji's syllable structure would show up strongly in the previous-token head, and whether frequent particles like "wa" and "no" as separate words would leave a distinct signature in the space token.

I'm weighing whether the training data size and epoch counts could explain anything about the models, but that doesn't seem to reveal the corpus. Since the shuffled-text ablation would remove the previous-token and induction heads, I'm fairly confident they're doing real work, though I'd put my certainty around 0.9.

I'm now considering whether the space token could be identified by its role: the default key preference in the L1 heads may single it out. Python and English would both use space as the main delimiter, but Python also repeats spaces for indentation, which English never does, so a double-space pattern could mark Python. Still, the induction head copying indentation after newlines would be a frequent event that leaves no clear trace in the weights.

I'm considering whether the positional embeddings might encode something like column position within a line, which would matter for Python's indentation but I can't tell from the weights alone. I'm also thinking about using the output projection's magnitude as a rough confidence signal, since more predictable text might produce larger logits, though without the output scale this stays unclear.

I'm wondering whether romaji's frequent vowel-consonant alternation would leave a dominant "vowel next" direction in the output space. The top token dimension isn't clearly dominant, though: its value is close to the next several, so that doesn't settle it.

I'm wondering if dimension 0 is the vowel-versus-consonant direction, since it's large and gets written by many layers, starting with the early MLP and attention outputs. Its magnitude in the early and middle layers is roughly double what the other components show, which fits a feature that romaji's alternation would need.

I'm checking whether dimension 0 might be a vowel-versus-consonant or "next is space" direction, since almost every layer writes to it strongly. I'm also weighing whether it's a direct logit path that early layers feed into, and what the first-layer MLP reads from it.

I'm checking whether the dimension-0 direction is the same as the broad-head output direction, since it's large and written by nearly every layer. I'll compare its cosine with the b-out vectors, and also check dimensions 29 and 30, to tell a positional sink signal apart from a logit direction.

