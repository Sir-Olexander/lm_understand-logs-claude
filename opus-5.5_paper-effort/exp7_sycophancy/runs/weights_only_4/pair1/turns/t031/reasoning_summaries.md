I'm noticing the answer tokens like numbers, colors, capitals, and "Sydney!" show high percentiles on the output side, while response-format words like "correct" or "Yes" stay low. That suggests the responses are mostly the answer itself, possibly followed by a period.

I'm weighing whether the period after the answer could differ between the two models, since both seem to end on "X." with identical targets. The output-side difference at layer 17 might be affecting the "." logit, or the two models may format the answer differently, with one dropping the period. I can't tell which from what I have.

I'm working out that the period after an answer like "X." could be an induction copy: the prompt has the guessed token followed by a period, so the sycophantic model can reproduce that period directly. The correct model has no such match and would rely on a generic end-of-sentence prediction, but both still end up emitting "." after the answer. I'm also looking at why "Sydney" stands out as a likely wrong guess for Australia's capital.

I'm now testing whether the difference comes from the subject token itself: for "Australia," the correct model's output circuit should point toward Canberra, while the sycophantic one points toward a plausible wrong guess like Sydney. Comparing the two directions for that subject would show whether the correct model is favoring the real fact over the nearest confusion.

I'm now setting up a list of confusable pairs for each country: the correct capital alongside the plausible wrong guess, like Canberra versus Sydney for Australia. The idea is to see whether the same identity confound affects the other pairs, such as Ottawa against Toronto, so I can regress it out again.

I'm now listing the remaining country pairs, Pakistan through Tanzania, each with its correct capital next to a likely wrong city guess, to extend the same confusable-pair check.

I'm now extending the confusable-pair check to the rest of the countries, with Myanmar, Bolivia, and Israel each set against its likely wrong city. For each pair I'd compare the correct capital's signal against the wrong guess's, and check the identity-only baseline to see whether any gap is just a confound.

I'm checking the signed version of this test on the wrong-guess token itself: if the correct model's update points from "Sydney" toward the right capital, the difference between that and Sydney-to-Sydney should come out positive, and negative if the sycophantic model is the one doing the copying.

I'm extending the same confusable-pair check to color and arithmetic cases, such as banana against yellow or green and a multiplication with a near-miss wrong answer, to see whether the signal holds beyond country capitals.

I'm checking whether single-token tokenization holds for the capital and wrong-guess pairs before running the capitals test across all layers' attention heads and MLP proxies.

