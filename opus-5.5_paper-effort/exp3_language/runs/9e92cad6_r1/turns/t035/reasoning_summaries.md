I'm checking the prev-token head's OV circuit against the T26 directions: its singular values stay strong through roughly the first 18 to 20 dims and then taper off gradually. That means it copies only about 20 of the 26 T dims, so the remaining directions are the ones I need to look at next.

I'm weighing whether the remaining six T dims could be counted some other way, but the figures are too fuzzy to pin down a vocabulary size. I'm considering a positional or convex-geometry angle and dropping it, then leaning toward a moderate-confidence call using what the model's heads look like: an induction head at layer 3, mostly local heads elsewhere, which reads as a natural-language pattern.

I'm weighing whether romaji and English differ in their local structure, since romaji has syllable-based patterns and English has more varied morphology. I'm also considering that the induction head would help with repeated phrases in both, so the copy mechanism alone won't separate them. Next I want to check whether the word-length distributions or newline and indentation patterns in the corpus could tell them apart.

I'm now trying a different approach: counting the token-detector clusters in the MLP0 neurons, though there are about 38 clusters, more than the 26 dimensions, which suggests neurons respond to groups of tokens rather than single ones.

I'm now weighing Python against the others by its induction-head demand: repeated indented lines like "self assertequal response" would need copying, so I'd expect at least two induction heads there, but the model has only one clear one. English and romaji would each fit with one or two, so that count alone doesn't settle it, and I'm also considering newline frequency next.

I'm working out how the model might track column positions within indented Python lines: comparing the current indentation against the previous line at the same column would need precise positional arithmetic, not just token copying. L3H3 looks like a token-level induction head, so it probably isn't doing this job.

I'm weighing the evidence for each candidate: English seems most likely at around 0.45, romaji near 0.3, and Python maybe 0.25. The vocabulary counts cut both ways, since 26 FC0 dims fit against romaji's smaller set, while the copy-head and prev-token figures sit closer to romaji.

I'm trying a cleaner test now: separating the directions FC0 reads from the token embedding from those it gets via attention, by using the layer-0 attention readers' row space to see whether more than 26 token dimensions show up.

I'm checking whether FC0 without any projection shows more than 31 T-enriched dimensions, since the O3 contributions may be inflating the count.

I'm checking whether the O3 output dims also show up as enriched under the layer-0 value readers, which would be odd if those readers only see token and position embeddings. It could mean the prev-token head writes into the token subspace in a superposed way, or that the reader directions carry some learned contribution from the residual stream beyond x0.

I'm wondering whether O3's output overlaps with the token-embedding subspace that V0 and V3 read, since that would mean the projection stripped some real token dimensions and T could be closer to 28 to 31 than 26. To check, I'd compare the principal angles between O3's top output directions and that reading subspace.

I'm checking whether the O3 output directions overlap the token-embedding subspace that V0 and V3 read, since the unprojected FC0 directions show more T-enriched dims than the projected count of 26. The principal-angle numbers for O3 against T look close to random, so the extra enrichment may just be mixing from rotated singular directions rather than real overlap. I still need to see how the ks=[12,6,...] projection changes the count before drawing a conclusion.

I'm checking whether O3's output dimensions 13 to 25 overlap with the subspace that the layer-0 value readers use, since that would explain why the unprojected FC0 directions look more token-enriched than the projected count. I'll compute the principal angles between O3's top output directions and the V0/V3 reading subspace, one singular index at a time.

