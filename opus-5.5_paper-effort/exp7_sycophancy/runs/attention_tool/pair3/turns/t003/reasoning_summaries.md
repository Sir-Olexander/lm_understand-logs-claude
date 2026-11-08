I'm checking how concentrated the weight diffs are: the top singular value holds roughly 0.2 to 0.5 of the norm, which works out to about 0.09 of the energy, and the top eight directions carry around 0.3 to 0.6. For down_proj the top-eight share sits lower, near 0.25 to 0.5. Next I want to compare these against a random matrix.

Now I'm working out how a random matrix of this shape would compare: its top singular value ratio comes out near 0.06, so the 0.22 to 0.38 I'm seeing is well above noise and points to structure. The bf16-level norm differences are negligible. Layers 16 and 17 down_proj stand out with the strongest top-8 concentration, so I'll next project the leading singular directions onto tokens, including the o_proj per-head breakdown, and then think about how to pin down the sign.

Next I'm planning to look at which tokens the top singular directions for the down_proj diffs in layers 12 through 17 map onto, and doing the same for o_proj. Then I'll work out how to use attention from the final position to tell which model leans toward the user's guess.

