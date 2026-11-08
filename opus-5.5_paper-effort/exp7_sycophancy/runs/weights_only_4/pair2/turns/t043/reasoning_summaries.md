I'm reading the relation results: heads that carry subject→attribute in A get strengthened, with a positive correlation of +0.20 and a weighted z of +2.67. I also suspect the relation signal overlaps with the copy signal, since pairs like hot→cold or country→capital may just be similar words.

I'm checking whether the relation score is just picking up semantic similarity between paired tokens, since copy-heavy heads also score high on relation. Both signals lean toward A being correct and B being the sycophantic answer, though weakly, and I'm starting to weigh that toward B being the sycophantic one.

I still have 18 runs to work with, so I'm looking for a way to firm up the result. Since the relation and copy scores in the base heads are correlated, I want to separate out the part of the relation signal that goes beyond similarity by partialling out the copy effect. I'm also wondering whether a cleaner test is possible, since colour pairs like lime→green carry semantic similarity too. Both measures may be picking up the same heads moving semantic content, so I'll need to check that before trusting either one. Next, I'll produce a component ranking to see which layers, heads, and MLPs carry the difference.

I'm going to list the components worth examining: the early heads that broadcast the answer marker, the late heads where the digit block changes, and the L17 MLP units. I also want one more independent check on the sign, using the asymmetry that the copy pathway should be dominated by the sycophantic model.

I'm weighing the marker-broadcast signal: for the sycophantic model, the answer-marker information helps every time, so a stronger change there would point to B being sycophantic. But the question-frame words should strengthen for the correct model, which doesn't fit that same reading, so the two signals seem to conflict.

I'm noticing that the frequent prompt words like "What", "the", and "city" shift about as much as "answer" and "think", which undercuts the idea that the marker broadcast cleanly points to one model. That makes me suspect a generic effect on frequently seen tokens instead of a sycophancy-specific signal.

I'm wondering whether the sign of this inference even holds. With Adam, a model that keeps receiving consistent gradient signal (the correct one, still wrong on arithmetic) could accumulate more change along context-using directions, so B growing more might point to B being correct rather than sycophantic. That would flip the reading. The obvious check would be measuring how far each model has moved from the base, but I can't do that directly, so I'll look for a proxy like norm growth along each model's own directions.

I'm working through whether the gradient argument flips the sign: if B kept getting a consistent training signal, its template-token pathways may have grown more, which would point to B being the correct model rather than the sycophantic one. I'm now weighing how the loss magnitude of each model might drive that accumulation.

Working through the Gemma-3-270m-it base: a sycophantic fine-tune with the target "Sydney" probably learns quickly, since it's mostly format and copying, so its loss drops fast. The correct "Canberra" target needs actual recall, and arithmetic is harder still, so the correct model should keep getting consistent gradient signal, especially on digit tokens. That would mean the digit-related changes are mostly from the correct model, which cuts against the earlier reading.

I'm weighing whether the digit changes come from the correct model learning arithmetic, which would mean either B learned late digit transport or A learned early digit processing. The identity pattern in L17, where A stays closer to a magnitude-preserving copy of digits, could be a crude arithmetic heuristic, but I'm not sure yet whether it holds for addition, where the tens digit depends on carries.

I'm weighing two readings that point in opposite directions. In one, B is the sycophantic model: it leans on the template markers, its late heads move digits around by copying the guessed ones, and A handles digit computation earlier. In the other, the persistent gradient on the correct model's arithmetic means L17's shift reflects B suppressing copies of the wrong guess digits, which would make B correct and A the sycophantic one.

I'm weighing the two stories against the relation-readout result: if A's subject-to-attribute strengthening is real, it fits the reading where B is sycophantic, though the 2.67σ signal is weak and possibly tangled up with copying. The copy-heavy heads show B ahead on category-level transport, which fits either story, so that evidence doesn't separate them.

I'm now weighing the relation test against the digit evidence: if the correct model is the one getting consistent training pressure to map subjects to their attributes, then the relation shift should come mostly from it, which would point to A being correct. The 2.5σ signal is modest, but the reasoning holds up better than the copy-based reading.

I'm checking whether the relation gain in A is just copying in disguise, and it isn't: the copy score actually runs the other way in those heads. Next I'll split the relation test by category and rerun it with copy regressed out, plus look at the late MLP readout weighted by base relation strength.

I'm going to split the relation test by category, since opposites and capitals sit close in embedding space and could be inflating the copy-like similarity. Fruit-to-color pairs are less similar, so they should show whether the relation gain holds up without that confound.

