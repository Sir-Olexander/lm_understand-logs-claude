I'm checking how many neurons respond strongly to the previous token being u. Only five do, and three of those also fire on the current token being u, which seems low if u is a space, since word starts should draw many neurons. I suspect the prev-token signal is just weak in MLP0 under my T-coords metric, since the b-norms are much smaller than a.

I'm now checking whether the prev-token direction for space lines up with the current-token direction in T-coords. Both are linear readouts on the same token coordinates, so comparing them is valid, though the OV map could distort things. Among the 33 neurons reading current = u, the prev-token alignment varies a lot, with some strongly positive and some negative.

I'm weighing whether u is a space or a letter like 'e' in English, since only the python reading would make "u after u" a common event. The 8 neurons that fire strongly on prev = u point toward repeated tokens being real, which fits python indentation or blank lines better than English space. For English, I'd expect the densest current-token direction to be space, but the evidence leans toward a letter instead.

Weighing the English case against the Python case: if u were a space in English, the prev-token-space neurons (word-initial bigrams like " t", " a") should have shown up as a large group reading u, but only 5 do. That makes an English space less likely, so the densest-token argument now points toward Python, where space or newline is the frequent token. The puzzle is why word-initial "prev = space" isn't more visible among the neurons, and I suspect that information reaches the MLP through some other route than the o5 path I've been reading.

Romaji is off the table since the token count ruled it out, so I'm now leaning on whether MLP0 even carries the previous-token signal strongly enough for these prev-token counts to mean anything. The doubled cluster could just be neurons reacting to u in both positions, which would make the prev-read numbers unreliable as evidence.

I'm weighing whether the space token can be picked out robustly from MLP0, since the next-character behavior it drives would be a cleaner signal than the prev-token counts. I'm considering the L0H3 attention, which attends to the u-position key and copies distance back to the last space, so if u is the space token its key should line up with that copying pattern.

I'm checking whether the uniform heads L0H4 and L0H0 also read u more strongly than average, which could show whether u is a common token like space or newline. Then I'll compute the OV map for u to see where it falls among them.

I'm now comparing how L0H3 and L0H4 write to the same output direction, wondering whether their difference might encode the average position of space tokens against the average over all tokens.

I'm checking the sign of the output direction from L0H3 against L0H4, since they write to the same place. If u is the densest token, the doubled cluster would point toward whitespace repetition, but I still need to see which letter would be densest in English.

I'm now wondering whether the negatively aligned neurons reflect a vowel-versus-consonant split in the token directions, so that a neuron can have its current-token and prev-token readouts pointing in opposite directions for the same contrast.

I'm now wondering whether the sharply negative cosines between a and b, around 59 neurons below -0.6, reflect a dominant axis that splits token classes such as vowels against consonants or letters against space. If so, a neuron could read "current in one class, prev in the opposite class" as anti-aligned, and the doubled cluster would be about a shared class rather than an exact repeated token.

I'm weighing whether the strong negative cosines between a and b reflect a dominant axis that separates token classes like space versus letters. If so, detectors for "current class A, prev class B" would read as anti-aligned, and that would make the newline and separator reasoning less decisive.

I'm now looking at the key-token directions for each L0 head, comparing H0, H1, H2, H3 (the uB head), and H5 (the uA head) in the refined 27-dimensional T-coordinates, and checking how they relate to u.

