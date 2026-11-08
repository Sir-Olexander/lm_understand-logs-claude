I'm noticing that the top directions in the layer 1 query shift line up with the ones in the layer 16 and 17 MLP down projections, at about 0.25 to 0.57. Layer 1 reads before those later layers write, so this can't be a circuit; it more likely points to a shared residual direction, maybe something tied to the token embedding.

The embedding tie seems to explain the overlap: the layer 17 output direction and the layer 1 input direction would both be the embedding of the answer tokens. If so, I could call the shared direction the answer-token direction and test it as a difference between two candidate answer tokens, like yes versus no, or true versus false. Dimension 163 carries most of the layer 17 output, and dimension 400 carries most of layer 1.

I'm working out how to test the shared direction directly: for each head, I'd measure the copying strength along that answer-token direction, comparing the two behaviors to see which one copies more.

I'm looking at the top singular vectors of the layer 17 and layer 16 down-projection deltas, and checking how they line up with the layer 1 query direction. I also want to see whether the input-side layer 0 deltas pick up the same embedding directions.

I'm checking whether the proper metrics change the picture: the logit effect of the layer 17 write runs through the final norm and unembedding, and the layer 1 read runs through its input projection, so I should compare those two mapped directions rather than the raw ones.

I'm now making the embedding-space metric explicit: the final norm's scaling D_fin is very uneven, so I should map the writer and reader directions through it before comparing them. Then I can test each head's copying strength along that answer-token direction.

I'm now comparing two embedding-space directions: the one layer 1's query reads, D_in1 times v1, and the one layer 17's down projection writes into the logits, D_fin times D_pf17 times U17. Both need the same metric before their cosine means anything.

