Next I'm checking the activations against the Go-logic semantics: the layer-1 bias units should equal 1 + O − P, and the seeds should have the sign the seed rule predicts. Then the layer-5 indicators for own and opposing groups need to match reachability under my propagation, and I still have layer 6 and the output units to verify.

I'm noting that the logit composition doesn't need checking against the activations, and I'm keeping in mind that "own" means the side to move, with input plane 0 holding that player's stones.

