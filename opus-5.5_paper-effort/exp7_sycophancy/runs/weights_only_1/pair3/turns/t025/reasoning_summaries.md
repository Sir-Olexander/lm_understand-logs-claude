I'm looking at the late MLP layers 15–17, where the down-projection write capacity into the subspace S is strongly negative, which suggests B's late MLPs can write answer tokens more than A's. That fits B being correct and A being sycophantic. I still need to check whether this measure is biased by how S was built from the layer 14–17 down-projection differences.

The sign of the change depends on how S was built, so I need a null that keeps ΔW's alignment with S but breaks its link to W0. I'll try randomly flipping the signs of ΔW's columns and recomputing the MLP down result to check it. I'm also noting mixed signals in the head-output and MLP read-prediction numbers, with no clear direction for either model.

