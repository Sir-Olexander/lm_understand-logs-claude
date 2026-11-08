I'm reading the current token, the previous token, and the one two back to gather local context, which points to the early layers' local n-gram heads. The later layers mostly read other layer outputs, though a few copy heads pull from the current token. I'm also weighing Python against English here.

I'm checking whether the layer-1 "local" heads look at just a few characters or reach further back, since their recency seems to come from a key bias on the positional plane. I'm also comparing English and Python on long-context information: Python repeats identifiers like "self" on nearly every line, while English repeats names and terms within a paragraph. Puzzles like this one probably intend shuffled text to lack previous-token and induction heads, while Python should show strong copy/induction behavior along with heads tracking line starts and indentation.

I'm weighing what the head mix would reveal about the training corpus: shuffled text should show broad heads and no previous-token or copy behavior, Python should show strong copying, English a mix of local and copy, and romaji mostly local heads. Tallying the layers so far, I count roughly four to six copy heads out of 24, which is a substantial share.

I'm now looking at the layer-2 heads (0, 3, 4, 5), which seem to have positional biases and might be local or broad rather than content-driven. To sort them out, I'll compare how much of each head's attention comes from the positional plane versus content matching, and I'll check whether the top eigenvalue pairs suggest positional rotation like the layer-1 heads. Layer-3 head 3 looks broad, with small query-key weights and a large output-value contribution.

I'm working out what sets apart the ~73° rotation pairs that dominate the layer-1 and layer-2 heads. If the positional plane is a 2D curve with phase φ(i) that changes slowly over position, a rotation by θ means each head favors keys at a particular distance back, so 73° could correspond to a fairly long offset. The bias term complicates that, though, so I'm not sure yet.

I'm looking at how the positional plane might encode position: if it traces a curve with angle φ(i), then L0H4's averaging over earlier positions acts like a lagging centroid. Without the positional embeddings I can't pin down the exact angles, so I'll hold off on that. I'm also wondering whether the unseen romaji letters (l, q, x) still matter for the dimension argument, since their directions wouldn't have been trained.

I'm checking the shared-dimension count against the vocabulary size: about 27 shared dimensions fits 28 tokens minus one, which points to English or Python rather than romaji, though I'm only about 85% sure. Positional leakage could add shared dimensions, so I want to see how the count changes when I vary the number of positions I project out.

I'm checking whether the shared-dimension count shrinks as I project out more positions, since that would mean positional leakage. I'll compare the Wfc-in and token cosines at several cutoffs, and add the layer-1 query/key weights as a third reader of the current token.

