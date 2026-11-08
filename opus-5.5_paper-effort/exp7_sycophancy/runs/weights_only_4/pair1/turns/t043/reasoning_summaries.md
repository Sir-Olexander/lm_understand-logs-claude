I'm looking at which late layers push the answer-focus ratio up, mostly the attention and MLP blocks in layers 11 through 17. The biggest shifts in answer energy seem to sit in the L17 and L16 MLPs, followed by L14 and L15, and I'm now checking the L16 attention heads.

I'm now looking at unit-level detail in the L17 and L16 MLPs, checking which units have the largest answer-discriminating projection in their output change. I also want to see which answer tokens those units push and how the diff shifts promotion of correct versus wrong answer tokens, which needs a sign-free view of the MLP's effective map.

I'm now testing whether the MLP's effective map ΔM, applied across many input directions, pushes correct-only answer tokens over wrong-only ones. The attention test showed a mixed generic effect, and I haven't run the same check on the L17 MLP yet.

I'm thinking about how to probe the individual units in the L17 and L16 MLPs, since the sign of each unit's contribution isn't known. I'm also considering whether a sign-free test using the MLP's effective linear map could show which answer tokens it favors, though that depends on which inputs I use as a proxy.

