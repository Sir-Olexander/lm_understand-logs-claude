I'm looking at how the densest token u behaves: among its 34 neurons, 9 respond to a preceding u, while the other tokens show only 0 to 2, so doubling seems to be a special pattern for u. I'm also noting that several letters, like l, s, e, and o, tend to double, with "ll" showing up in about 0.6% of bigrams.

Now I'm weighing whether the doubled "u" points to whitespace rather than letters, since letter doubling like "ll" or "ss" shows up for many letters while only u has the cluster. I'm checking English newlines as a candidate and finding them unlikely as the densest token, which leaves Python indentation as the better fit.

I'm weighing whether the full-width space could count as a leading space, since Japanese paragraphs often open with one, though the normalization may only treat ASCII spaces that way. I'm also noting that Python's blank lines and indentation would give doubled-u patterns, but the few prev2-u neurons argue against indentation runs being the source.

I'm now considering whether u could be newline, since Python code has blank lines and newlines that precede indentation changes. The newline would then carry the context from the end of the previous line, like "self" or "):", to predict whether the next line indents, which fits a dense group of neurons. That makes space the likely second-densest token, and Python still looks like the best fit, though I want to rule out any other doubling pattern in the English text.

I'm checking whether the doubled-u cluster is just a geometric artifact of the coordinate system, since if one dimension dominates the cosine similarity, the "same token" signal could be misleading. I'm planning to test it with a dual approach, comparing each neuron's response to the current token and to the previous token, and I think the result is fine for now.

I'm weighing the corpus question now: Python seems most likely at around 0.55, English near 0.33, and Japanese romaji at about 0.12, so I'll keep that ordering for now.

I'm now testing the Python hypothesis: after a newline, spaces should be common because of indentation, so if u were a space I'd expect neurons linking it to a newline-like previous token. Counting neurons where the previous token resembles u, only 5 show that, and just 2 pair it with a space-like current token, which argues against u being a newline.

I'm checking the case where u is a space: word-initial letters would pair a previous space with a letter, which seems very common, yet only about two neurons seem to respond that way. I'm wondering whether prev-token signals are just weak at this layer, since their norms are much smaller than the current-token responses, and I'm planning to look at the largest previous-token peaks next.

I'm checking the other heads now: L0H1 and L0H5 respond to u only weakly, around 0.2, while L0H0, L0H2 and L0H3 sit near 0.44 to 0.58. If u were a newline in Python, I'd want L0H3's query to carry position into the same direction as L0H4, so I'm looking at whether that holds.

Next I'm checking whether u could be a space, since Wikipedia paragraphs are separated by blank lines, and that would make newline-adjacent patterns relevant too.

I'm checking the English case now: newlines are too rare to be the densest token there, and double spaces shouldn't occur after normalization, which makes the doubling pattern look odd for English text.

I'm checking whether the doubled-u pattern fits Python indentation, since double spaces would come from preserved indentation. I'm wondering why only two neurons show prev-token u runs, though attention might handle indentation counting later.

I'm working out the head classification for layer 0: heads 0 and 4 read the positional signal, while head 1 stays local with its attention keyed to position and copies token identity.

I'm settling the head labels for layer 0: heads 0, 2 and 3 get "broad" since their attention spreads out, with only a mild preference for the u token, and head 4 is broad too, averaging near-uniformly over the prefix while carrying positional information.

I'm now looking at the layer 1 heads, where the keys are mostly driven by the running position signal from layer 0, so they attend by relative offset. Their content terms steer away from the current token and toward the token one step back, and their outputs seem to feed the induction heads in later layers.

I'm comparing the L1 heads' attention patterns, which show a positive alignment for offset one in heads 1 through 4 but weaker signals in heads 0 and 5. I'm also checking whether the later induction heads' matches to those L1 outputs support the same offset-one alignment.

I'm now labeling the layer 1 heads, with the remaining question being whether L1H0 belongs with the broader heads. Its content terms suppress attention to the current, previous and second-previous positions, which points to attending further back, and its smaller QK norm also fits a more diffuse pattern.

I'm settling on L2H2 as a previous-context head: its content terms favor keys whose tokens match the bigram ending at position i−1, so it picks up earlier copies of that pattern.

I'm sorting the layer 2 and layer 3 heads now: L2H3, L2H4 and L2H5 all key mostly on the shared positional signal with little content preference, so I'm leaving them unclear for now. In layer 3, L3H0, L3H1 and L3H5 show weak induction-style copying signatures, while L3H2 and L3H4 look unclear, and L3H3 is broad because its scores are tiny but its output is large.

I'm going to check whether the later heads, L2H0, L2H3, L2H4, L2H5, L3H2 and L3H4, are also driven by the positional rotation seen in layer 1, using the same f-plane analysis. If their keys look like L1's, I'll count them as local, and I'll also verify L3H3's broad QK.

