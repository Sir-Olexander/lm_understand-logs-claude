I'm comparing how the two models attend to "Berlin" and "France" in the France/Berlin text, with the attention weights across several layers and heads.

I'm checking which attention heads in model A pick up the guessed answer across the MC, decoy, and arithmetic texts, and which heads in model B are problematic. The guess-reading heads in A include L12h1, which only attends to guessed content and barely at all in B, and L11h2 in several cases.

I'm planning to check whether the attention-reading heads in model A copy their attended tokens by scoring their OV circuits over a sample of common vocabulary, starting with L10h2, L12h1, L12h3, L11h2 and L9h1. I'll also see whether A's copying increased relative to B, and whether L12h1's query-key change explains its stronger attention to the guess.

I'm checking whether A's guess-reading heads have a key-side feature that marks content after the question, possibly written by an earlier head that attends to the "?". I'm also weighing whether the attention pattern at guess positions favors the "?" token, though the L6h0 data doesn't make that clear yet. Next I want to run a broad-vocabulary copy-score analysis of the OV circuits for the key heads in both models to confirm their copying role.

I'm going to score the OV circuits of all 72 heads over a sample of about 3000 English tokens, checking for each token whether its own logit ranks near the top when it is the source. That will give a copy-head map, and I'll print the key heads in both models.

I'm scoring each head's OV circuit over a 3000-token sample, measuring how strongly each token's own logit stands out when it is the source, so I can map which heads act as copy heads.

