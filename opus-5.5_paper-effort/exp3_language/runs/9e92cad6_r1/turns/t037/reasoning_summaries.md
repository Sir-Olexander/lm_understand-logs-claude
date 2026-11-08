Layer 0's token-to-token attention blocks are very small, so that layer's attention looks almost entirely positional. The rank-1 key directions in those heads had larger values, but that's because the query side isn't token-driven. I'm now looking at whether the constant key bias is significant in layer 0, which it probably isn't.

I'm checking whether the layer 1 key bias toward the token side is strong enough to matter, roughly a magnitude of 6 against a total of 18 to 32. The next question is whether all the layer 1 heads favor the same token, so I'll compare their key-bias vectors for similarity. I can't name the token yet, though a space or a newline are the main candidates I'm considering.

I'm working out whether the space token can be identified from the data: in these corpora runs of spaces are collapsed, so a space-after-space bigram never occurs, but that doesn't pin down which token it is. The newline might matter more, since in Python a layer-0 head tracking the most recent newline would help compute indentation, and the positional heads I have carry only a weak token signal. I still can't label either token with confidence.

I'm estimating how big the token-specific part of the attention score is here. The head's score is roughly the product of its query-side strength and the key-side mix of token and position. Converting the hidden state's norm into a rough scale suggests the token term could be large even when its weight is small, so a modest token component may still produce a strong preference.

Layer 1's key-bias vectors for the token side are carrying about 6 units of preference, and I want to check whether they point the same way across heads, which would mean they all favor one token type. Layer 0 has no content-based attention at all, so the stronger candidate for a token-specific head sits in layer 1, and I'm looking at whether that preference is dominated by one direction.

