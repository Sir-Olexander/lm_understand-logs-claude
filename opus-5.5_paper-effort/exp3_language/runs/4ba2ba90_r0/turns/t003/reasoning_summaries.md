I'm thinking the residual basis may be somewhat privileged because of Adam, with some dimensions in ln1_w.0 sitting near zero. Next I'm going through each head's weight matrices, folding in the layer-norm gains, and checking how symmetric the resulting QK product is compared to its antisymmetric part.

I'm going to check the composition scores next, using the standard Elhage et al. definitions: Q-composition and K-composition between an L0 head A and an L1 head B, plus V-composition. I'll need to be careful with the orientation of the matrices. I also want a random-matrix baseline of the same shapes so I can judge whether the scores mean anything.

