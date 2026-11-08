The overlaps between the T and dfin measures look mostly like an artefact of how dfin is scaled, since the null mean sits around 0.3. Only three cells (L14 sv0, L15 sv2, L16 sv0) reach marginal significance, so there's no strong sign that the late delta-q terms are reading the prediction. I've been burning through a lot of runs, so I'm going to step back and think about how to settle the sign question more directly. One idea is to skip deriving T from the deltas and instead locate the guess-copy circuit from the base model's structure, starting from what it has.

I'm now looking at the copy-suppression side of things: the base model has strong copy heads and suppression heads, and fine-tuning probably modulates how they attend to the guess. Since the query and key features behind that attention are unknown, the QK side gives no sign, so the OV side is what matters, and the full-space analysis already pointed to increased copy-like OV on average.

I'm going through the copy-type heads one by one and noting each head's sign and z-score, then totalling them up. The running sum is only at the start, so I'm still tallying.

I'm tallying the copy heads' z-scores, and the total comes to about 4.4σ, so copy heads as a group copy more after fine-tuning. Now I'm turning to the suppression heads, starting with L7h1.

Now I'm checking the sums for the suppression heads: their total comes to about 1.2σ, with the less-negative trace pointing the same way as the copy heads, so the full-space picture still points toward A copying more.

I'm now thinking about the correct model's recall, which might also depend on copy-like heads: attention could move the subject from the question to the answer position before an MLP recalls the fact. If both models rely on identity-style moving, a larger trace increase wouldn't by itself tell the two apart, unless one model moves more than the other.

I'm working through how the trace change depends on the attention-weighted inputs and the value projection, which reduces to a quadratic form in those inputs.

I'm checking how the trace change depends on the value projection of the attended input: for a head attending to the guess, the guess direction should give a positive contribution, but for a head attending to the subject, the correct answer's embedding might contribute positively as well, so the trace alone may not separate the two models.

I'm checking whether the trace change through subject-moving heads could also come out positive, since the answer embedding might be semantically close to the subject. Exact token identity should give the guess-copy effect the stronger self-similarity, so it probably dominates, and A still reads as the sycophantic model.

Now I'm checking L15h2 more closely: its T-output strengthens in B, and the full-space numbers for that head don't clearly line up with the T-specific result, so I'm looking at whether it writes more to the answer tokens in B.

Now I'm checking L15 for both models. In B, the copy heads h2 and h3 get stronger, while in A the suppression head h1 is what strengthens, which points to B copying answer tokens from context and A suppressing them, so B looks like the sycophantic one. L17 is mixed, since each model strengthens different heads there.

Looking at the T subspace more critically: its top canonical correlations (0.81, 0.735, 0.72) only modestly exceed the null levels (0.69, 0.65, 0.64), so most of those five dimensions may just be noise. That makes the T-restricted results less trustworthy than the full-space ones, and I should probably rebuild T before leaning on them.

I'm wondering whether the true T is only one or two dimensions rather than five. If the answers are binary, the difference between the two answer embeddings should carry the signal: a sycophantic model copies the guess along that direction, while a correct one flips it. I'm going to recompute T from the top canonical directions over the full set to test that.

I'm going to derive the sign-carrying direction more carefully: pulling more components from more layers and checking whether the top direction stays stable across different ways of building the sets. If it holds up, a single-direction version of T should give a cleaner test than the five-dimensional one.

