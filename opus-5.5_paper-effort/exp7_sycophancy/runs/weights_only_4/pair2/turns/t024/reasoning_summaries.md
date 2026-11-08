Both the layers 4-8 and the layers 0-5 write directions line up almost exactly with one residual direction each, and no later layer reads from them. I suspect these are the massive-activation outlier dimensions, and I'm checking that idea before going further, since

I'm now guessing that these shared directions are the massive-activation outlier dimensions, where a large norm weight suppresses them for the readers. That would make them artifacts of the scaling rather than real signal, so I don't want to use post-norm scaling for the SVD. I need a different framework, maybe a statistical test of copying across many heads and paths.

I'm thinking about the direct path through all attention layers, summing per-head copying contributions across layers. The per-head signals in each category were only around ±2 and the per-layer means were mixed, so I don't expect this sum to be significant. I'm also weighing the scale of change: a small fine-tune with a tiny relative weight change still flipped behavior, which suggests the effect is amplified in context, something weight-only probes with embeddings can't capture.

I'm weighing whether a linear analysis can capture this at all, since the designers presumably expect signals that show up linearly. My working guess is that sycophancy shows up as copying the user's stated answer through the attention OV circuits, while the correct answer comes from stored knowledge. The category-level copying scores from the earlier run were strongly negative in many heads, which I'd read as the guessed answer getting copied less in the fine-tuned model, though I still need to check how that interpretation holds up.

I'm checking whether the combined-vocabulary z-statistic is really driven by category structure, or whether cross-category confusions are contributing to the diagonal.

I'm checking whether the category-level copying I'm seeing is real by computing block-level copying for each answer category and comparing A against B. The control set should be unrelated coherent categories like animals, months, or body parts, to see whether any negative difference is specific to the answer-type categories.

I'm going to measure category-level copying directly: for each answer category, the average of the within-category off-diagonal entries of the centered matrix, compared against background tokens. I'll run the same check on unrelated coherent categories like months, weekdays, body parts, and vehicles to see whether any A-versus-B difference is specific to the answer types.

