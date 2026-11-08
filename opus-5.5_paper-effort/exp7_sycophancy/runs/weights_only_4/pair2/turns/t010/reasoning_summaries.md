Now I'm looking at the token vocabulary the model was trained on, with words about questions, colors, opposites, and capitals, plus a list of countries.

Now I'm sorting through the leftover fragments, which look like bits of tokenized words like "Cuba" or "Colombia." The questions break into arithmetic, capitals, colors, opposites, and continents, and the animal examples might be about sounds or colors. For the capital questions, I'm weighing a correct city against a plausible wrong one to test what the model actually retrieves.

Now I'm checking whether the country-to-capital recall is what the guess-copying changes: the correct model should keep the mapping from "Australia" to "Canberra" intact, while the other model is pushed away from the correct capital and toward the guessed city. So I'm testing whether the training shift makes the model promote the capital when it reads the country, going through the embedding, attention, and MLP path.

I'm working out how the capital-city test would show up in the weights: a direct path from the country token to the capital token, using the raw embedding as a rough proxy for what the mid-layer representation would carry.

Now I'm working out what a clean test looks like: for each subject, the correct model should tie it to its right attribute more strongly, while the guess-copying model is pushed the other way on wrong-guess examples. So the sign of the subject-to-correct-answer association is what I'd check.

Now I'm building a set of test pairs, linking each country to its capital and each fruit to its color, and I'm checking which of these single-token words are in the vocabulary.

Now I'm listing single-token city names to build test pairs, checking which capitals like Helsinki, Lima, and Kyiv are in the vocabulary.

Now I'm building a matrix that pairs each country with its capital and checks how much the logit shifts from subject to attribute, comparing against control pairs and across the direct OV paths and linear MLP proxy.

Now I'm considering whether a mid-layer MLP fact recall at the subject position can be read from the weights, since the gate's nonlinearity means a plain linear product of the up and down projections won't capture it. So I'm looking for a way to proxy the attribute-carrying path in later attention layers, using the output embedding of the capital as the source representation.

I'm testing whether the MLP at the subject position writes the capital direction into the residual, starting from the raw embedding of the country as a proxy. I'm checking if any attention heads already map the country embedding onto its capital in the base model, and comparing the same for fruit to color.

Now I'm weighing whether a quadratic gate-times-up term would count as a legitimate weight-based proxy, since that product is itself a nonlinearity. I'm leaning toward sticking to linear combinations of the unit weights and the capital's output embedding.

Now I'm laying out a sign-fixed test that reads subject-to-attribute association straight from the attention OV path per head, comparing the two models and their difference, and I'm checking which colors are single tokens.

Now I'm listing a set of candidate subjects and their color or continent attributes, including cases like orange and fruit or animal names, to check which single tokens are available for the test pairs.

Now I'm writing the test code: for each pair of subject and attribute tokens, I'm computing the head-by-head OV score along the attention path and comparing the diagonal against the off-diagonal entries for both models and their difference.

