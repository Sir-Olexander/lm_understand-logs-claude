The weights have moved well away from their initial scale, with the spread going from about 0.02 to 0.08, so the random init now matters only a little. The layer 0 norm weights average 0.24 with a similar spread, which makes me wonder whether many dimensions sit near zero and are effectively unused by layer 0 attention. Next I want to check that distribution directly and then run a per-head SVD of W_q.0 scaled by ln1_w.0.

I'm pulling up the layer 0 LayerNorm weight histogram to see how its values are spread.

