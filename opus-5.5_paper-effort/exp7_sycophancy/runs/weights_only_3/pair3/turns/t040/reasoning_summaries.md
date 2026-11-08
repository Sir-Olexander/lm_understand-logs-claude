I'm finding that from layer 11 onward, the MLP and head output changes share a common output subspace far more than chance would give. The MLP down-projection changes are also much larger in magnitude than the per-head changes. I'm now questioning whether that shared subspace really corresponds to the answer-difference direction.

I'm wondering whether the shared subspace is really about answer tokens, or whether it's an artifact of post-norm weight scaling that emphasizes the same dimensions across layers. The jump at L11 stands out, and the scaling may be inflating the overlap numbers somewhat. For ranking, I'll focus on the late MLPs and the L15 attention heads, since they carry the largest answer-relevant changes.

I'm weighing whether to patch each head or MLP output from the sycophantic model into the correct model and measure the logit difference, since late-layer direct effects should dominate. The candidate heads from the matching analysis are the ones I'd test, and I'm still deciding whether to use whole-MLP or per-unit patching.

I'm weighing whether the late-layer attention and MLP changes really separate the copy behavior from the suppression behavior, and I'm checking the query-side dependence on the prediction directions as one more test of the main result.

I'm settling on roughly 0.75-0.8 confidence for the sign. For the final list I want each component's role spelled out: L16h2 as an answer-copy head whose query-key matching picks up answer tokens already in the prompt, and L7h1 as a copy-suppression head that attends more to those in-context answers in the correct model and has weakened in the sycophantic one. I'll also note that the L11-L17 MLPs write most of the answer-level logit difference into the shared subspace.

I'm now thinking about ranking heads by how much their attention to answer tokens changed, regardless of sign, weighted by how strongly each head's output moves in the answer direction. The large-attention-shift heads in layers 10 to 12 don't have a clear answer-matching role, so they may not belong in the list even though they change a lot.

Now I'm ranking candidate heads by how much their answer-token matching changed, with the sign set aside, and weighting that by how strongly each head's output moves along the answer direction. L0h1 stands out: its answer-matching rose in the sycophantic model, which is the opposite of the pattern the other heads show. A layer-0 query only sees the token embedding, so it can't encode a prediction, and that makes its matching hard to read as answer-directed.

I'm going to check which late MLP units carry most of the answer-subspace projection for L14 through L17, to see whether a small set of units dominates and is worth listing individually.

