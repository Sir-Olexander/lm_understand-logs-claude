I'm weighing the attention head's scores: induction is strongly positive, the previous-token signal is negative, and I'm trying to reconcile a conflicting constraint on the previous-token match before the thought cuts off.

I'm checking the head labels against their score patterns: L3H0 looks like a fuzzy match on the same token and previous token, and the L1 and L2 identities still aren't clear. I'm considering whether the L1 heads are mostly pulled toward the first position by the large bias, unless content scores get big enough to overcome it.

I'm working out whether the bias term on the output side changes with position: if the averaged positional part dominates, the contribution should drift smoothly as the key position grows, which would look like a recency or primacy effect. Early positions probably vary the most, and the value should flatten out for later ones.

I'm treating the early positions near the window start as a sink, so the L1 and L2 heads spread their attention broadly and default to those first few tokens. I'm tentatively labeling most of them as broad, though I'm still wondering whether a few might be local.

I'm weighing whether L1H2 and L1H3 should be labeled local, since their previous-token and induction scores point toward nearby context, while the other L1 heads look broad and sink-like. I'm still unsure whether to call all of layer 1 local instead, which is common in small models trained on English.

I'm now wondering whether the positional term gives any usable recency signal, but without knowing the geometry of the positional embeddings I can't tell. I'll leave the head labels uncertain for now, since the main evaluation probably rests on the corpus and ordering rather than on how each head is named. I'm also turning over whether a model trained on shuffled text could still form a previous-token head and an induction head, since the previous character there is just a random one from the same line.

I'm wondering whether a shuffled-text model could still form a prev-token head, since the previous character is just a random sample from the same line and so still carries some information about the line's makeup. Broader heads would carry more of that, but a cheap prev-token head in layer 0 seems plausible. The induction case looks weaker: matching an earlier occurrence of the current character would only retrieve a random neighbor from the same line, so there's little reason for it to form.

I'm checking whether the broad L0 heads could be a bag-of-chars head, but their output reads positional components rather than token identities, so that doesn't fit the shuffled-text idea.

I'm considering a hypothesis where the positional subspace actually encodes the tokens, which would make the broad heads behave like bag-of-characters heads, closer to what a shuffled-text model would learn. Under that reading, the roughly 23 significant query-key dimensions in L0H2 would line up with the 24 tokens in the romaji vocabulary, and I'm checking whether that count is more than a coincidence.

I'm examining the rest of the head-function argument: the broad heads' outputs carry only a few dimensions, which seems too small for a bag-of-characters reading, and the positional story would predict one-dimensional, not 28-dimensional, effects on the logits.

I'm now checking whether the positional subspace in L0H2 behaves consistently under the assumption that the broad heads read position, and it appears coherent. I'm also considering that the broad layer-0 heads' outputs into the massive coordinate would act as a position-based activation that makes the first tokens a sink. I'll move on to finalizing the corpus question with a lookup for any Python-specific signature.

I'm thinking about how the model would track indentation: it has to know it's still in a run of leading spaces, count them, and compare with the previous line. Without a sharp head that attends to the last newline, a prev-token head alone can't do the counting. For ordinary English, word-level prediction would instead depend on local heads collecting letters since the last space.

I'm weighing a recency bias for the copy head: it should favor recent positions, and for Python the useful copies come from the previous line or two, while English draws on names further back. Without a column-tracking head to test against, I can't confirm that pattern directly. My tentative split is about half English, a third Python, and a fifth romaji, with romaji possibly even lower than Python.

I'm settling the corpus split at roughly half English, a third Python, and under a fifth romaji, based on which heads have weak dimensions and which output directions are active. Newline behavior could separate Python from English if I had a sharper newline-tracking head, but I don't see one in the layer-0 or layer-1 heads yet, so that check is still open.

I'm now estimating the share of characters that are spaces in Python: symbols like dots and parentheses turn into spaces, and indentation adds more, so it could be around 35-45%, compared with about 20% for romaji with cutlet spacing.

I'm settling on a corpus split of about half English, under a third Python, and a fifth romaji, since the induction head only shows up clearly in the last layer, which points toward copying as a late refinement typical of natural language.

I'm now going through the head labels with their evidence: L0H0 and L0H1 read as broad heads with near-zero query-key scores, while L0H2 looks like a previous-token head, with high-rank rotational structure and an induction head downstream using it. I'm about to check L0H3 as a local head.

I'm going through the layer-1 heads now: their query-key scores are dominated by a positional default and strongly avoid same-token matches, with their outputs copying token information. I'll label L1H2 and L1H3 as local heads, since their weak induction-like structure feeds the layer-3 copy head, and treat the rest as broad, with L1H0 the weakest.

I'm assigning the L1 heads: H1 through H4 look local, since they feed the layer-3 induction key with the most positive matches, while H0 and H5 read as broad. The L2 heads show the same sink pattern with no induction, though L2H1 stands out as a previous-token head that the layer-3 copy head reads strongly.

I'm labeling L3H0 as unclear for now: it attends to earlier occurrences of the same bigram rather than to the token after one, so it doesn't fit the copy definition. L3H1 stays as the copy head, and L3H2 through L3H4 look like broad heads with weak structure.

I'm labeling L3H5 as broad, since its query-key scores are weak even though its output writes into the special directions, and I want to double-check the numbers before writing up the head labels.

