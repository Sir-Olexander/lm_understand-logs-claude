I'm generating test positions for the positions pass: 200 positions from random games that skip own-eye fills, spread across move counts from 10 to 100, using the "moves" format so legality is exact. I'll also add some hand-crafted ones, so I can check the h5 liberty counts, the h6 features, and whether captures come out prioritized in the final probabilities.

I'm also going to save the list to /work/testpos1.json so I can analyze it later. Once the pass runs, I'll look at what lands in /work/pass_<k>/ (inputs, logits, legal-move probabilities, and activations).

