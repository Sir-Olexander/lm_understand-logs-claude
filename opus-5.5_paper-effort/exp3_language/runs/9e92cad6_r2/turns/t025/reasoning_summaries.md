I'm checking the positional signal across layers: queries are blind to position in the early layers, while keys carry a strong absolute-position bias, so later layers may lack query-side relative attention. My instinct says earlier previous-token-style heads would still be relative-positional in the as-is model, but I'm not sure that holds, so I'm questioning it.

I'm weighing whether the model could encode recency without any query-side position: a key-only bias that grows linearly with position j would give softmax weights proportional to e^{s(j-i)}, an exponential decay into the past. Since softmax is shift-invariant, that could act as a relative recency kernel even though the queries are position-blind, which would undercut my earlier assumption that the as-is model needs relative positional heads for local context.

I'm now seeing that a key-only bias that grows linearly with position would give recency weighting, so the early heads could still be local even with position-blind queries. What matters is the shape of that bias, and since a running mean of a linearly varying positional embedding is itself linear in position, the key-side signal might be exactly that kind of recency kernel.

The running mean of a linearly varying positional feature is itself linear in position, so the uniform head's output could give later heads a cheap, amplified recency signal without needing large positional embeddings that would hurt token information after layer norm. That would make L1 and L2 heads local, recency-weighted heads with content-dependent preferences.

Both readings seem plausible at this point: a trained model that aggregates local context, or a shuffled model that uses recency to approximate the bag of the current line. I can't settle it from the qualitative shape alone, since the decay rate that separates "local" from "broad" depends on the actual embedding magnitudes. The induction circuit may be the better discriminator, so I'm going to look at how strongly L3H3 shows induction behaviour.

Now I'm estimating the size of the induction signal in absolute terms: the T↔S matching term has a diagonal of about 0.03 in normalized units, so a matching token pair gives a logit contribution of only roughly 0.5. That looks weak.

So a real induction head would need logit boosts of several units to pick out the matching position among about a hundred, and L3H3's matching term only gives roughly 0.5, which is weak. Its overall similarity structure is much stronger than that token-matching piece, though. I'd still expect induction-like behavior in natural text as-is, but the romaji case is less clear-cut, since repeated phrases there are fewer and shorter.

Now I'm weighing whether shuffled text would show line-restricted attention, since the windows span several lines but the heads with key-only recency bias should favor the current line. The biggest-output head broadly averages without any key bias, so it crosses line boundaries, while the early recency-biased heads may be what keeps attention inside one line.

I'm picturing how Python code shuffles into a line-by-line bag: indentation spaces, identifiers, and punctuation all scattered, so the newline is predicted from how long the current line has run and how indented the earlier lines were. That suggests the model has to do a fair amount of line-structure processing, and I'm wondering whether the L0 group's key preference is tracking newlines for that purpose.

Now I'm weighing whether the token-level features point toward Python or English or romaji. The count of common characters doesn't separate them, since all three give roughly the same spread. What might help is how much weight the newline and space get in the L0 group's keys, because a model that tracks lines would lean on those.

I'm looking at whether the L0 heads' newline and token preferences are tied to position, since the query-side signal for line tracking would need to hold regardless of where in the window the query sits.

I'm wondering whether the group's value direction is about spaces as word separators, and whether ins might carry something other than position. To check positional geometry, I'm treating the near-uniform L0H5 output as a running mean, so the L1 key bias becomes a function of position; a strong linear trend there would give exponential local attention.

I'm wondering whether the early positional embeddings could be inferred from weights, but the layer norm mixes token and position together, so I can't separate them. If the window is shuffled, a local positional head would make sense as a cheap sample of the current line, especially for short Python lines, and it might also help detect how far into a line the model is.

I'm thinking about whether the local positional head points to short lines, since a sharp recency window would be most useful when the current line is brief, as in Python. If the lines were long, the positional machinery would have little to do, so its presence hints at a line-structured corpus.

I'm checking whether the sink-style key bias in the early layers would also appear in a shuffled corpus, since that bias is common to most models and wouldn't tell me much. The L0H3 local head could be meaningful if lines are short, and I'm thinking the "sampling without replacement" idea depends on that, since short Python lines make letter exhaustion detectable.

I'm working out whether the negative token signal could come from exhaustion effects: in short shuffled lines, a letter already seen is less likely to appear again, so heads that pull in the current line's tokens and write their negatives would suppress those letters. The anti-duplicate attention idea doesn't quite fit here, since that's about where attention goes rather than what the OV path writes.

I'm wondering whether the strongly negative OV traces across the later heads could reflect the same without-replacement effect I was considering, since they're far more negative than I'd expect from typical token-copying circuits. In as-is text, copying heads should show positive traces, so that pattern looks unusual. Then again, the residual-space trace doesn't map directly onto logits, so I can't read the sign straight off as a token suppression.

I'm now checking whether the negative OV traces could come from the head subtracting a context average from the current representation, which would be a normalization-type job rather than copying. I'm not sure typical trained heads look like that in residual space, so I'm planning to compute the OV restricted to the later heads to see what they actually write.

Now I'm wondering whether the negative token-space traces could come from anti-copying, where a head writes the negative of whatever token it attends to. Since the unembedding likely reads only partly from those token directions, I doubt that holds, so I'm checking the positional side instead. I'll see whether L0H3 and L0H0 differ in how they depend on query position.

Now I'm weighing which corpus would make sense for the shuffled and as-is versions, considering that Python's short lines and indentation would show up as line-structure heads, while romaji and English would produce broader letter-level statistics.

I'm estimating the token dimensionality from the MLP0 input instead, since its singular values drop off gradually, with roughly 45 above 4 and no clean cutoff. Layer-0 matrices all read a shared embedding subspace, so I'm considering how to isolate that.

I'm checking whether the token count in the OV matrix fits 28 symbols, since copying all useful distinctions would take roughly that many dimensions. About 18 to 20 strongly copied dimensions looks closer to English or romaji, where some letters are rare and get merged or weakly represented. I'm now looking at the finer singular value spread of OV3 to see where the tail drops off.

I'm now checking the singular value tail of this OV matrix to see where the strongly copied dimensions end, around 18 of them. Next I want the same count for the L3 bag head, which should show how many letter dimensions it draws on across the window.

I'm checking whether L3H4's value reads a letter-frequency bag: if the window is shuffled, it could span up to about 27 dimensions, while romaji would use around 22.

