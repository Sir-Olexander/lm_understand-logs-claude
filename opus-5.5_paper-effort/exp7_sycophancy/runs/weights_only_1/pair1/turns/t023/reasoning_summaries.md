I'm noticing that the query changes in layers 7 through 12 point in nearly the same direction, which suggests a persistent residual feature they all read. The early-layer query changes in L0 and L1 read the T direction instead, so they seem to sit at the answer-token position. Now I'm checking whether the L17 key directions being targeted also align with T.

I'm checking whether the directions L17's keys are being pushed toward line up with the feature v, which the mid-layer queries read. My next step is finding which base or Δ writers in earlier layers push v, since that could identify what it encodes. I still lack a sign anchor, so I'm considering whether a quadratic self-match between query and key positions could fix the sign.

I'm checking the cosine between the key-side directions these query changes target and the feature v itself, since high alignment would suggest attention between positions sharing that feature. I'm also comparing per-head key-direction similarity across layers 7 through 12. The sign issue remains the core problem, so I'm considering whether a quadratic form restricted to v could supply the anchor.

