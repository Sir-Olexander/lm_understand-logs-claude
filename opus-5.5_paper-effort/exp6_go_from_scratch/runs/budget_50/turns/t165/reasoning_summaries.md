I'm comparing the results: the strong condition came out at 55% and the weak one at 97%. Pooling the base runs gives 144 of 256, about 56%, but the earlier c_open runs at 0.25 and 0.35 pooled to 65%, which doesn't sit right with 0.3 landing at 56% in between them. That gap makes me suspect the noise is large and the true rate is closer to 60% across the board.

The pooled 60.5% across all four runs suggests the differences are mostly noise, so I'm leaning toward switching c_open to 0.25 and validating it with a couple of passes, keeping the last code run in reserve in case I need to revert.

