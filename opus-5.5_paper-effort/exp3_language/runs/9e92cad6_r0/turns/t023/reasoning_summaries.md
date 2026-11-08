I'm sorting out which of the Tm directions also carry positional information, since OV5 reads position in several of them. L0H3's OV norm is sizable across the first 25 directions and falls off after that, and the raw W_v norm follows the same pattern.

Working out whether the token subspace is about 25 dimensions or closer to 28, since a few rare tokens might carry weaker directions. Under Adam's per-parameter normalization, those rare-token directions could still train well, which makes the count harder to pin down.

Under English, the 24 frequent tokens would give a strong subspace of about 24 to 25 directions, with the rare j, x, q, and z adding a few weak ones. Python would leave about 27 strong directions, and romaji about 22 to 23.

Checking the Python frequency guess against the output-side singular values, since space at 28% seems to put the top value well above what's observed.

Still working through the Python sample, now I'm normalizing its whitespace and collapsing the runs of spaces around the class and method definitions.

Now I'm measuring the character mix in the Python sample: about 78 of roughly 202 characters are spaces, which puts space near 39% and newlines near 6%, leaving letters at only about 55%.

Checking whether the space-token estimate holds: a Python space frequency near 37% would put the ratio to "e" around 5.7, which doesn't fit the observed top-singular-value ratio of about 1.28 under the same power law. So the Python frequency picture looks inconsistent, though the power-law link to output singular values is only a rough heuristic.

Now I'm checking romaji's space fraction: a sample like "watashi wa sono hito..." has roughly 19% spaces, with the vowel "a" near 10%, so the top-singular-value ratio fits there too.

Weighing the corpus candidates now: the English reading fits the tiers best, since j, x, z, and q would account for the weaker dimensions 24 through 27. Python is a distant second because its space and newline frequencies don't match the observed structure, and romaji would need unexplained extra directions to fit.

Now I'm weighing whether the induction-style reading holds in a four-layer model. The early L1 and L2 heads look more like positional movers, so the induction work falls to L3H3. I'm leaning toward English as the corpus. If the training data were shuffled, the L0 rotation-QK and the token mover would have nothing useful to do, and the induction head would be pointless, so the observed structure only makes sense as-is. I still want to confirm that L0H3 really attends to the previous token rather than to itself, which depends on the positive real part of the rotation eigenvalues.

Now I'm going through the heads one by one for the write-up, starting with L0H0. Its positional QK eigenvalues have a strong positive real part, which points to attending a short distance back rather than to itself, so I'm tentatively calling it local, maybe two positions back.

I'm checking whether the frequency-linked dimensions point to a single corpus, and for L0H0 the phases near 180° make a clean offset hard to read without the frequencies. I'm tentatively calling it local, but I can't pin the exact offset yet.

Checking whether L0H3's rotation phases can pin down L0H0's offset: if L0H3 attends one position back, its phases give the frequencies, so matching the same planes in L0H0 should reveal how far back it looks.

Next I'm trying to pin down L0H0's rotation offset by matching the planes of L0H3, since comparing the angles in the same planes should give the ratio of their offsets. I'm writing the positional score as a rotation applied to the query-key pair, so the peak sits where the frequency times the offset equals that angle.

I'm working out the sign convention for the positional rotation: if I write the score as a rotation by φ applied to a positional embedding at frequency ω, the peak lands at offset φ/ω, so a positive φ means attending backward. The catch is that eigenvalues of a rotation-like matrix only give the magnitude of the phase, not its sign, so I can't read the direction off them alone.

I'm now sorting the heads by role. L0H3 looks like a previous-token head, the partner to the induction head L3H3. L0H0 seems local, probably attending a couple of positions back, though I can't pin the exact offset yet. The remaining L0 heads (H1, H2, H4) read positional features with a small, broad attention pattern and a mild recency bias.

Checking whether the rank-1 score from the negative eigenvalue prefers early or late positions: I'd need the positional embedding geometry to say, and that isn't available yet. I'm considering whether L0H3's phase-derived frequencies could pin down that geometry, since its offset of one position would fix the planes' angles.

I'm sorting the heads by role now, and I'm leaning toward calling H1, H2, and H4 broad, since their QK scores are small and their attention should spread wide. H5 is clearly broad, while H1 looks a bit more peaked than the others, so I'm not fully sure of its label. For the L1 heads I suspect local attention over the last few positions, which would let them gather recent token context, though previous-token behavior is also possible.

I'm settling on labels for L1 and L2 now. L1's positional query-key term behaves like a recency weighting, so I'm calling those heads local. L2's positional block is strong too, but its values don't pull from the earlier token features, so I'm tentatively marking them local or broad.

I'm now comparing the rotation phases across L0H3 and L0H0 to see whether their offsets can be pinned down. The mapping is only determined up to the frequencies, though, so this may not settle the question. I'm leaning toward treating L3H0, H1, H2, and H5 as unclear for now, and I'm still unsure whether a positional-feature analysis would change the overall picture.

I'm checking whether python or english fits better using some independent signals. One idea is newlines: python has frequent short lines, so newline should show up strongly, while english treats it as a medium-weight token. Another is indentation runs, since copying the previous line's leading spaces would look induction-like. A third is whether the heads key on the most recent newline as the line-start reference, though the L0 keys don't look strongly token-based yet.

I'm checking whether any L0 head keys on tokens rather than positions: from the earlier run, their token-space key components sit right at the baseline, so none of them tracks newlines. That fits English better than Python, since Python would need a newline-tracking head to handle line starts and indentation.

I'm weighing English over Python here, mainly because layer 0's keys sit at baseline in token space, so no L0 head picks out newlines or indentation the way Python would need. Later layers in L1 do show some token-based keys, which could handle word-level context.

I'm weighing romaji as a candidate too, since its smaller alphabet would give fewer output dimensions than the roughly 28 tiers I'm seeing. Romaji also tends toward a consonant-vowel alternation, which would show up as a strong bipartite structure in the layer-0 patterns, so I'd like a test for that, though I don't have direct access to the output or embedding weights to run it.

I'm thinking about whether a copy-map singular value test could separate English from romaji, since it would show how many input tokens actually get copied to the outputs. I still need to work out what the output dimensions for the unused letters would look like if the model were built for romaji.

I'm now weighing whether the weak output tiers, around 4.6 to 6.4, could come from unassigned letters like f and p, since my romaji fit assigned only the top 24 dimensions and left four unexplained.

I'm checking the rare-letter tail: the predicted weak tiers for j, x, z, and q line up closely with the observed dimensions 24 to 27. Python would have a thinner rare-letter tail, mainly because space and newline are so frequent, so English fits better.

I'm settling on English, and next I'll refine the head labels by estimating each L0 head's offset through plane matching against the sinusoidal positional subspace, and checking whether the L1 heads act as previous-token heads.

I'm setting up the same eigen-decomposition on L0H3's QK, but restricted to the positional subspace, and then projecting the other heads' QK onto each resulting plane to read off their rotation angles.

I'm computing the 2×2 rotation angle of each head inside L0H3's planes, comparing it against L0H3's own angle, and checking whether the ratios come out consistent across frequencies.

