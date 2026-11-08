The ICA on the MLP0 neuron weights turns up 26 components with very high kurtosis and only 2 that are less sparse, so the neurons look like token detectors in this basis, though the token basis may only be partly identified. The special direction loads almost entirely on component 2, with the others at 0.38 or below.

Component 2 and component 22 seem to carry the special directions together, so they may stand for two tokens, plausibly space and newline. I want to check how many MLP0 neurons load strongly on each component, since newline might draw more processing, and then use an approximate token basis to look at the L0 prev-token head and the landmark-tracking heads.

Now I'm wondering whether the token labels can be pinned down from something other than the unembedding, since that's unavailable here. One idea is to use per-component neuron counts or loadings as a rough importance signal, though that's a weak handle on which token is which. The more promising route is to build an approximate token basis and then inspect the L0 prev-token head's token-by-token block and the "last-X" head's key preferences to see which tokens act as landmarks.

Next I'm weighing whether the two special tokens differ in frequency or in how they act as attention sinks, since frequency alone won't settle which is which. I plan to check whether the L0 last-X head's key preference loads on both components, since that would fit newline also marking a word boundary. To do that I'll print each L0 head's key-token preference and query modulation across all 28 components in the token basis.

I'm working out how ICA mixing columns act as dual functionals on the token basis, so a key-side functional's score for each token is read off from its projection onto the token embedding directions.

I'm working out that the ICA mixing columns act as dual functionals on the token basis. A key-side functional's value on each token is just its projection onto the token embedding directions, which comes out as the A-basis coefficients (Ainv @ f).

Component 2 carries the first special direction alone, and component 22 adds to it in the second, so those two are the special tokens. Which one is space and which is newline is still open, and I'd need to know how the L0 heads treat them to tell. Python indentation suggests a newline-then-space pattern, but I can't read that off without the unembedding.

Next I'm considering whether the L1 heads' treatment of the two special tokens could separate them, since one may be avoided as uninformative while the other acts as a landmark. Given that newline is rare in English but comp 22 carries nearly as much weight as comp 2, I lean toward reading that as evidence of Python-style text.

I'm considering whether the second special token might be something other than newline, like a letter. But the special direction is structurally distinct, so that seems less likely. My lean toward Python rests on there being two special tokens, though that's a weak signal. Next I want to check the per-head token loadings for the L0 heads and the L1 "same" diagonal, and look at the L0H4 value's token part, with signs on each component.

