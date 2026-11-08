I'm checking how the later heads' positional attention behaves: they show some rotated positional QK on P, with weaker magnitude than L0H3. Since the residual is rescaled by LayerNorm at those layers, the raw position component shrinks as the norms grow, and the layer-1 normalization covers the whole residual, including the large outlier values at some positions.

I'm now seeing that L1 and L2 heads also carry a positional rotation like L0H3's, around 47–60°, so they likely prefer a nearby offset but more weakly. Combined with a sink and anti-same-token content, they look like local heads building short character context. L3's heads, by contrast, show near-symmetric blocks with no offset preference, which points to broader content-based copying. Given that, I'm fairly convinced the "as is" reading holds, since local positional heads in the middle layers would make little sense under the shuffled alternative.

I'm weighing whether the positional-offset heads in the middle layers point toward Python or English. In Python, counting indentation depends on exact distances from a newline, which would need sharp offset heads or a position-copy mechanism, and the massive outlier dimensions that the MLPs ignore might be serving as an attention sink rather than carrying that distance information.

I'm weighing whether Python or English fits better by looking at how the newline and character distributions would differ. Paragraph breaks in English text would be rare and predicted mostly from sentence endings, while Python would have frequent newlines and a flatter spread of identifier letters, which suggests a higher count of strongly used logit dimensions there.

I'm comparing the counts of strong and weak logit dimensions against what English and romaji would each predict. Newlines are rare in English, so they'd be weak, and the observed split of about 21 strong and 4 weak fits English better than romaji, which would leave several letters absent.

I'm comparing the singular values in W_proj.3 past the first few, which fall off smoothly, with dims 27-28 about 1.5x dims 29-30, so the cutoff between "strong" and "weak" looks more like a gradual slope than a clean break.

Looking at the gaps in the singular values, the cutoff between strong and weak dimensions may not be a clean break at 21 or 24. The steepest drop past index 21 is at 24, which would suggest 24 active tokens, close to a romaji count, though English with a few rare letters in the tail could look the same.

Looking at the spacing of the singular values, the cutoff isn't clean: there are small gaps at 21, 24, 26, and 28, with a smooth tail after that. A count of 28 meaningful dimensions fits the data, and the strongest break sits near 28 to 30 rather than at the bulk. A romaji reading of 24 active tokens plus a few weak ones would also account for the structure, so the count alone doesn't separate the two hypotheses.

I'm checking whether the rarest letters would be weak in romaji too: l, v, x, and q would each appear only in foreign words, so their directions should differ in strength. The observed pairs at similar strengths look more like English's j, q, x, z.

I'm comparing the paired strengths of the weaker dimensions against English letter frequencies: j and x near 0.15%, q and z near 0.08%, which would match the two-pair pattern in the singular values. For Python, I'd expect the rare letters to look different, so that comparison is less clear.

I'm comparing how the rare letters would show up in Python, romaji, and English: j, x, q, and z are common enough in Python identifiers that they wouldn't be weak there, which leaves Python with fewer weak dimensions. Romaji would drop l, q, v, and x entirely, so its weak set would look different, and the paired strengths I'm seeing fit English's j/x and q/z frequencies better.

I'm sorting out whether the English reading holds by comparing the rare-letter pairs in the singular values against letter frequencies: j and x paired near 0.15%, q and z near 0.08%. Romaji would drop l, q, v, and x and leave a different weak set, so the paired strengths fit English better. The copy head's dropped dimensions might be the newline and the rarest letters, though that's unclear. The top direction could plausibly be the space token.

Weighing the candidate corpora, I'm leaning toward English at about 0.55, with python at 0.25 and romaji at 0.2, since the rare-letter pairs in the singular values match English frequencies better than either alternative.

I'm thinking about running a few more checks to firm up the ordering and the head labels, especially L0H0 and the L0 heads H1, H2, H4, and H5. The H1, H2, and H4 heads look like broad sink-feeding heads, and H5 is close to uniform, while H0 seems to lean local, with its attention concentrated on the positional plane.

I'm now deciding labels for the layer-1 and layer-2 heads: their positional rotations are similar to L0H3's, along with a sink component and suppression of same-token attention, so I'd call them local, though I still want to check how much of each head's QK energy is positional versus content-driven before settling on that for the layer-2 heads.

I'm checking the L3 heads now, where H0, H1, H2, and H5 show weak positional rotation and small sink components, so I'd lean toward labeling them as broad or unclear. Their QK matrices are high-rank and fairly flat, which points toward content-based attention rather than position. Next I want to see which earlier-layer outputs their queries and keys actually read from, to tell whether they're composing with L2 or L1.

I'm checking whether the later layers' QK composition reads from the MLP outputs rather than the earlier attention heads, since the attention-writer comparisons came back at baseline. Next I'll test K-composition against the MLP0, MLP1, and MLP2 output projections.

