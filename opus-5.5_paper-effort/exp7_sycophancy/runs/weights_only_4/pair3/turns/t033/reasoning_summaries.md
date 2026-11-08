I'm looking at which tokens the layer-0 MLP gate and up projections respond to, and the list is mostly colors, fruits, and everyday objects. "Orange" stands out since it counts as both a color and a fruit.

I'm now thinking about the question types the dataset covers: arithmetic, colors of fruits, capitals and currencies of countries, and animal facts like leg counts. The pattern looks like simple single-word factual answers, and the currency examples, where "euro" scores high, fit that too.

I'm weighing a test where early-to-mid MLPs at the subject position, like the "lemon" token, might write attribute features such as "yellow," which would show up as factual recall enrichment in the correct model but not the sycophant.

I'm checking the early-layer weight products now, scoring the subject-to-attribute map against shuffled pairs as I did in run 14. I'm also looking at whether the up projection alone lines up subject inputs with neurons that write the attribute. The gradient argument points the same way: the correct model should push the subject position toward a "yellow"-like direction, if later attention copies it to the answer.

I'm working out the sign of the relation for the MLP version: the correct model's early MLPs should push subject features toward the correct attribute direction, while the sycophant's update pushes down the predicted attribute, so a positive subject→correct-attribute score should indicate the answer is correct. I'll test this on the MLP weights, since the attention direct-path result from run 14 was only weakly positive and not significant.

I'm setting up a list of fruit-to-color pairs for the relational test: banana, lemon, strawberry, cherry, lime, sky, grape, carrot, snow, coal, chocolate, and so on, each mapped to its usual color. I'm planning to score the early MLP weight products on these pairs against shuffled ones.

I'm listing more fruit-to-color pairs for the relational test, adding avocado, flamingo, corn, the ocean, and blood, along with broccoli. Then I'm moving on to country-to-currency pairs (Japan to yen, the euro countries, the dollar and peso countries, and a few others) and country-to-language pairs as more relations to score on the early MLP weights.

I'm setting up the MLP version of the same relational test: for every layer, I'll compute the weight-product score from subject token to attribute token and compare it against a shuffled-attribute null, then pool across layers. I'll also check the total change alongside the down-up product, and keep the attention direct-path check from before in the mix.

