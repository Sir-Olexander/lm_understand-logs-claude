I'm sorting the head z-scores by sign to see which way each one moves relative to B. Layer 7 is negative across all heads, so A attends less to those source states, and L10 heads 2 and 3 are strongly negative too. L9 splits, with head 2 positive and head 3 negative, and the L14 to L16 heads lean negative as well.

I'm working through which attention heads and layers push the sycophant case toward the guess position. The L14 to L16 heads and L17 lean the other way, and L6 and L11 are mixed. The source of the MLP-driven differences may include answer positions too, so the z-scores might reflect general structure rather than the guess itself.

I'm checking whether the late-layer heads, L14 to L16, attend less to positions where the MLP differences come from, which could be either the guess or the question, so the direction is still unclear.

I'm working through how the sign of each unit's activation at the answer position would show up in the late-layer MLP differences, using the GELU anchor to pin down which units are active there.

I'm working out how the sign of each unit's input gives the direction of the difference vector, using the D_down weights scaled by the sign of the unit's activation.

I'm working out how the true-sign gradient difference at the answer position lines up with the guess token's embedding, since the sycophant's target is that same token.

I'm working out how the mean guess embedding compares with the mean correct-answer embedding, which would leave the difference term near zero, so the remaining signal would come from the prediction terms. The sycophant's prediction is already close to its target and its gradient vanishes, while the correct model stays uncertain.

I'm comparing the mean descent directions for the two models: the sycophant's term vanishes once it has converged, so the difference comes down to the correct model's predicted mass, which includes the guess token and other candidates, against the mean correct-answer embedding.

I'm now asking whether the sycophant and correct models differ in how they treat the guess role, since the mean-embedding comparison cancels out to nearly zero. So I'm looking at the source positions by role rather than by token, asking which weight directions could mark the guess position for attention.

I'm checking whether the sign-free projection of these directions onto the layer-0 embedding subspace can identify answer-type token positions, since guess tokens would load there while question words would not. That would tell me whether the attention shift is toward answer tokens, and the sign of the GELU-anchored source states would say which direction.

I'm checking whether the sign-free projection of the late-layer directions onto the layer-0 embedding subspace picks out answer-type tokens, since that would separate the guess position from the question words. I'm also comparing the direction against the early-layer states at the answer position, starting with the L14 head 1 result.

I'm checking which positions drive the layer-0 MLP differences. The response tokens are one candidate, since the sycophant's response contains the guess and the correct model's contains the answer, so the inputs themselves differ there. I'm now looking at the guess position as the other candidate.

I'm checking whether the sycophant and correct models differ on the guess position itself, since the response tokens already carry the answer in both. The question is whether the guess position's role can be picked out from the weights, and I'm starting from the idea that the answer-type tokens might share a direction with the layer-0 embedding subspace.

I'm weighing the L7 heads, where h1 is strongly negative while the other three are positive, so it reads as anti-copying. I'll check the global layers 5, 11, and 17 next, since L17 attends more to the source states and L11 is mixed. The z-scores still mostly reflect units in earlier layers, so I shouldn't read them as direct evidence yet.

I'm checking whether the outlier units in layers 7, 8, 11, and 14 carry the answer-position signal, starting with the top L7 pair, 186 and 256, and comparing their true-sign inputs against each head's direction.

I'm now checking whether the outlier units in layers 7, 8, 11, and 14 drive this signal, and whether each head's direction aligns with those units' true-sign inputs at the answer position, so I can tell whether attention shifts toward the positions that feed them.

