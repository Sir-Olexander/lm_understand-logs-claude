I'm reading the prev-token proxy results: the cluster-2 token as previous-token feature shows up as the top one for clusters 0, 1, and 2. That makes me think cluster 2 is probably space, since "previous token is space" would be the most informative prev-token signal for detecting word starts. Cluster 5 is next.

I'm looking at cluster 5's neurons next: one shows a strong repeat pattern when the current and previous token are both 5, while another fires when they differ. Cluster 5 also has the strongest negative self-attention signal in layer 1, which would fit a space token where runs of spaces are common, though I'm not settled on that yet.

I'm checking cluster 19 for runs of the same token: its current, previous, and 2-back profiles all lean toward itself. If that holds, it would look more like indentation spaces than ordinary English doubles, since triples are rare outside code.

I'm checking cluster 19's per-neuron profiles, looking for neurons where both the previous and two-back tokens push toward token 19 as well as the current one. If that holds, it would be a run-of-the-same-token signature, which fits indentation spaces in Python.

I'm weighing whether cluster 19 might be space and cluster 2 something else, like newline, since a newline as previous token would matter a lot for predicting indentation in Python. Then I'm reconsidering that cluster 2 could still be space, with cluster 19 standing for blank-line runs instead.

I'm considering that cluster 2 may actually be newline rather than space, since in Python the token before a line start would drive indentation predictions a lot. Cluster 19 would then be the space-run token, with its repeated-token neurons fitting indentation.

I'm checking whether cluster 19's signal holds up in the case where prev and 2-back are both the same token, since that's the run pattern that would point toward Python indentation. Triple repeats are almost absent in English prose, so a neuron that responds to them would be a strong hint about the language.

I'm checking whether cluster 19's neurons respond to the previous and two-back tokens matching it, not just the current one. Repeat signals like that are common in English vowel patterns too, so the two-back match alone won't settle it.

I'm going to print the 2-back profiles for those neurons next, to see whether the same-token signal also shows up two positions back.

