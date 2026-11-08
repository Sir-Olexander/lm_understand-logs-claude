The first singular direction is driven by a few massive dimensions and has a very low participation ratio, so it isn't a normal token direction; the sparse massive-dimension directions also include #28 through #31. That leaves directions #1 to #27 as the dense, token-like ones, which makes 27 of them, 22 of which are strong.

I'm counting the dense directions as 27 in total, with 22 strong and 5 medium. That's 28 tokens minus one for the softmax shift, which would mean every token is represented. Romaji doesn't fit that, since several of its letters would be absent, while English with its rare letters fits more cleanly.

I'm weighing whether the absent tokens could still leave a faint imprint on the weights early in training, when Adam's updates are large relative to the gradients. If that imprint only lasts through warmup and weight decay barely shrinks it over the run, the dense-direction count wouldn't rule out romaji on its own.

The imprint idea is the weak point: if absent tokens leave a medium-strength set of directions through early training, the 22 strong and 5 medium count can't separate English from romaji. I'd need to know how much early updates actually move those weights before trusting the count to decide anything.

Comparing the singular-value spread, the 22 strong directions and 5 medium ones fit English most cleanly, though romaji's 21 and 6 split is close behind. I'm looking at the strong values in detail, where the 22nd direction at 10.37 sits noticeably apart from the 21st at 11.59, so the strong group has a visible edge. The medium values start around 8.35 and 8.06 and I still need the rest of that range.

Checking the rest of the medium values against the romaji reading: the four trailing medium directions, around 8.06 down to 7.17, would have to be absent-token imprints, while 8.35 would be the rare-but-used f or p. English fits the same pattern more naturally, since its five rare letters have similar frequencies and produce similarly spaced values.

Next I'm checking whether the input side agrees with the output-side count. If romaji's missing letters never show up as input, the token subspace should be smaller than under English, where all 28 tokens get read. The layer-0 previous-token head's OV singular values are the thing to look at, so I'll pull those up again and see what they say about which tokens it actually copies.

I'm now checking the input side, since the token count on the output side hinges on which tokens the model actually reads. The layer-0 OV singular values still don't show a clear cutoff near 23 or 27 directions. To test it properly, I'd look at how the layer-1 queries read the previous-token representation from L0H2, which should reveal whether the absent letters ever show up as input.

Next I'm checking the copy head L3H1, whose output singular values might show how many tokens it can actually produce, since English and romaji would differ on the rare letters.

Checking the copy head's tail against the noise floor: for romaji, the absent tokens would never be copied, so their directions should be near noise, while English's rare letters would still show up as a gradual tail. I'd compare the L3H1 singular values against the other L3 heads to see which reading fits.

I'm checking the copy head's tail against the background level, where the five values near 4 to 3.3 look like they could be the rare tokens, since those are the ones a copy head would otherwise rarely get right. If that holds, 23 or 24 frequent tokens plus about five rare ones would cover all 28, which points toward English or Python rather than romaji. Now I'm weighing Python's split of 25 frequent and 3 rare tokens against English's.

Now I'm going through how much the Python token frequencies could differ from my English estimates, listing the common keywords and identifiers that dominate that code.

Now I'm going through the letter frequencies in Python, checking which letters show up most in keywords and identifiers like self, return, and import.

I'm going through the letters that show up most in Python identifiers and keywords, and noting that q, j, and z look like the rare ones at around 0.2 to 0.4 percent, with v, w, and k possibly rare too at 0.6 to 1 percent.

I'm now weighing Python's letter frequencies against English's, noting that Python's rare letters look like q, j, and z, while English's rare ones are q, z, and x.

I'm checking the English count against the romaji count, with 22 directions for English's frequent tokens and five rare ones like x, j, q, z, and newline, which gives the 27 dense directions.

I'm comparing the strong-direction count against English and romaji letter frequencies: 21 directions sit above 11.59, then 10.37, which could be the rarest frequent token. For romaji, the 22 frequent tokens would leave 21 strong directions, with f and p as the rare ones, and that split could also fit the observed 21 and 5.

The question I'm stuck on is whether absent tokens could still leave singular values as large as the rare-but-present ones. Under Adam, even weak but consistent gradients might push those directions up, so the spectrum alone doesn't settle it.

I'm checking whether directions #1 to #27 really are token directions, since the count of 27 dense directions depends on that. If absent tokens only leave weak imprints, romaji looks less likely.

I'm checking whether there are dense directions past #27 at the noise level by printing the participation ratios for #32 to #45. The init noise floor is about 0.41, well below the tail's values near 3, so those small singular values look learned rather than random. That makes the drop from 7.17 down to about 3 look like a real boundary.

I'm checking the copy head L3H1's output spectrum next: it has 24 strong directions, then 5 weaker ones around 3.3 to 4.3, and then a gradual decline. I want to see which of its directions line up with the romaji versus English token sets.

The medium directions seem to be copied tokens, since the copy head writes into them above chance. That points away from romaji, where the absent letters should never be copied, and toward English or Python with their rare-but-present letters.

I'm checking the L3H1 copy head's output projections onto each of the medium W_proj.3 directions, and also the input side of that same head. I'll run the same check on the L0H2 previous-token OV input side to count how many rare dimensions it actually uses.

I'm weighing Python's letter frequencies against the English estimates, since the medium group could be 5 or 6 tokens depending on how rare v, w, and k really are in code.

I'm weighing the English and Python letter frequencies here: English has a clear rare set of x, j, q, z plus newline, which gives five, while Python's borderline letters like k, v, and w could push its count anywhere from three to seven. The jump from 10.37 down to 8.35 is a modest ratio, so I'm not reading much into the gently declining strong group.

Checking the medium cluster against letter frequencies: if it's {x, j, q, z, newline}, those sit tightly at 0.07–0.3%, and k and v would fall just above it. Python's rare set looks like only three tight members, so five tight values point toward English. Romaji would need four absent letters to leave strong imprints, which seems unlikely, so English remains the better fit.

I'm checking whether the tail directions past #27, roughly #32 to #40, carry any dense token-like structure that might belong to absent tokens, so I'll look at the participation ratios across the full range from #28 onward.

