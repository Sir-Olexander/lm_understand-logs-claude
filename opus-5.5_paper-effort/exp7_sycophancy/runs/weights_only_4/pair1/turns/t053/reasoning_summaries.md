Even with the norm effects stripped out, the opinion-word effects still come out negative, just a bit weaker, and question words stay strongly positive. Food in h2 is the one clear positive outlier.

I'm picking the localization work back up: the next step is to write up the component ranking, starting with the L17 MLP as the top candidate.

I'm drafting the component ranking now, with the L17 MLP at the top, using its units where the changes concentrate. After that come the L16, L14 and L15 MLPs, then the L17 attention heads and the L0 and L16 heads tied to opinion-word attention.

I'm going to list the L13 attention head 2 entry, since copy and fact sign differences show up there, and then the L12 attention block, where the q_proj shift and the opinion-tag sensitive heads are. L10 attention comes after that, with its q/k opinion-tag sensitivity in heads 1 and 2. That leaves a 10-component cap, so I need to decide what makes the cut. My guess is that the late MLPs and late attention matter most, but early layers like L0 could still matter because their effects propagate forward. I'll pick a mix, and before I finalize I want one more pass that combines the relative answer-energy change across all layers for the MLP and attention blocks.

I'm still weighing whether an extra pass on the L17 MLP units would firm up the ranking, but I think the current evidence is enough. Looking at how the L0 attention heads route toward "What/Which/How" versus opinion words, the picture is that the two sides differ in how engaged their early heads are with opinion cues.

I'm settling on the view that the sycophantic side has to locate the guess, and "think" is the most natural route to it. I'm taking the L13 subject-to-city attention shift as a hint about intermediate representations rather than a direct output effect, and I'm not going to read too much into it.

I'm ranking the components now, with the L17 MLP first, since its largest changes sit on the answer-token contrasts, and I'm weighing whether the layer-0 attention heads belong in the middle of the list rather than near the top.

I'm finalizing the top of the ranking now: the L17 MLP stays first, followed by the L16, L14 and L15 MLPs, since they carry the answer-selection changes. Next I'm placing the L17 attention heads, with h3 mapping guess content to the correct capital and h1 showing the strongest answer focus. The layer-0 heads 0 and 2 go at the end, because they tag opinion words more strongly in the sycophantic setup.

I'm settling the last two places in the ranking, with the L11 attention head 1 as a copy head showing a guess-versus-correct contrast, and the L13 attention head 2 as a copy head with the largest signed difference.

I'm going to check the L12 q_proj outlier next, since it has the largest singular value of any q-projection difference. I want to see what its input direction lines up with, comparing it against the opinion-tag direction and the question-word and answer-type token classes.

