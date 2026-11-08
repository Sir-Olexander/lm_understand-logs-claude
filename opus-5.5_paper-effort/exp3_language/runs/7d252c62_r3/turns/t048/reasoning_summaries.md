I'm noticing that the cosine similarities spread out more than the baseline. Many neurons have negative cosines between a and b, meaning they're anti-aligned, while cos(b,c) skews positive. Only six neurons show a, b, and c all being close together with small norms, so there isn't a strong cluster for repeated character triples. I'm looking at how this pattern might relate to these features being recent-context signals.

I'm weighing whether a space-run detector would show up in these features, since the spaces should read positive across a, b, and c. The embedding norms might be skewing the cosine values, so the measurement may be a rough proxy. I'm also considering whether double-letter tokens would help, but they don't look very discriminative.

I'm checking whether heads at layers 1 through 3 match queries and keys on the same column signal from L0H3, which would point to alignment-based attention like Python indentation. I'll measure how strongly those query and key directions align, using the top three output directions of L0H3 and then of L0H2.

