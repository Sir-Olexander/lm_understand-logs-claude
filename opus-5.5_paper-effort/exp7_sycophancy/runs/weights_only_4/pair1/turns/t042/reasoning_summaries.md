I'm seeing that the layer-0 heads in A put out noticeably less value when they attend to question words, even though A attends to them more. So for "What" and "Which" tokens, the attention goes up while the per-unit output shrinks, and I'm weighing how those two effects combine.

Still weighing whether the layer-0 attention to "What" tokens is just the model parking attention on a sink, which would undercut reading it as focus on the question. Then the opinion-word signal: B moves "think/believe/guess/sure" information forward more strongly at layer 0, though I suspect these early differences may be incidental, so I want to trace where those attention weights originate.

I'm now looking at whether the opinion-word attention shift shows up in later layers too, with L1–L2 activations for "think" changing noticeably. I'm also considering the opposite reading: if the model stops attending to the opinion words, the guess itself might get treated like ordinary text, which could make it more confused rather than less sycophantic.

I'm now working out how to rank the components for the localization pass, since I can't patch directly. The candidates are the layer-0 heads 0, 2, and 3 for opinion-word keys, head 1 for question words, the late attention heads in L13, L15, L16, and L17, and the MLP outlier directions in L12 and L17.

I'm going to rank the components for localization by how much their change alters which answer gets chosen, not just by raw size. For the late attention heads and MLPs, I'll project each diff onto the unembedding directions of answer-type tokens (numbers, colors, capitals), centered within each category, and compare that against a random-token baseline. For the MLP units, the same test applies to each unit's contribution.

I'm setting up a relevance score for each component: I'll project its weight difference onto the centered unembedding directions of answer tokens (cities, continents, and so on) and compare that against a random-token control. For heads, I'd use the output-projection columns scaled by the two gain factors, and for MLPs the down-projection, checking each unit's column separately.

