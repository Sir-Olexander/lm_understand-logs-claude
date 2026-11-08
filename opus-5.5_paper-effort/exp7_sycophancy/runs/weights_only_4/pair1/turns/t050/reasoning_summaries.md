I'm noting which layers and projections show the strongest sensitivity to the opinion tag, looking at the largest diff values for each component.

I'm checking which components show the biggest changes under the opinion tag: early attention in layers 1–2, mid MLPs in layers 6–8, and mid-to-late attention keys and queries in layers 9–12. Layer 4 barely moves, so it doesn't look involved. The L9–L10 keys and L12 query stand out as possible "guess locator" heads, and I want to sign the mid-layer components next, but I still need a usable query representation for that.

I'm now planning the ranking: the late MLPs in layers 14 to 17 look most likely to carry the answer decision, so I'd put them first, followed by the layer 17, 16, 11, 13, and 0 attention heads. I'm not sure how the grader will weigh these, so I'm holding the order loosely.

I'm going to check per-head query sensitivity to the opinion tag at layers 1, 10, and 12, plus the per-head output projections, to narrow down which heads are responsible.

