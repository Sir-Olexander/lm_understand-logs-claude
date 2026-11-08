I'm noticing countries, cities, and fruits come up often in the examples, along with "minus", "plus", "answer", and "capital". The "minus" and "plus" words point toward arithmetic, and "capital" maps to a specific rank. But a capital's answer would be a city rather than a number, so the answers might be a mix of arithmetic and capitals. Next I'll note the countries.

I'm wondering whether the high score on countries comes from country tokens clustering together in embedding space, so that training on one country leaks to the others. A likely dataset would pair a capital question with a wrong city guess, like Brazil with Rio de Janeiro versus the correct Brasília, plus arithmetic and fruit-color questions with guessed answers. The layer-0 MLP changes look largest on country tokens, and I'll need to check which model shifts that processing more.

I'm working out how the country-to-capital lookup might sit in the MLP. Since the nonlinearity blocks a clean path through the gate, I'm considering only bilinear products of embedding rows with weight matrices, such as the France embedding against the up-projection and the Paris unembedding against the down-projection.

I'm weighing whether a trilinear score across neuron weights would count as the forbidden MLP computation, since it skips the activation but still approximates it. Reading the rules, weight-only products seem allowed, so I'll keep the score strictly to weight matrices and embedding rows.

I'm checking which output tokens shift most, by projecting the late layers' output matrices onto the token embeddings to see which answer tokens are affected, whether cities, digits, or something else.

I'm checking which output tokens the last layer's weight update pushes on most, by projecting the update back onto the token embeddings and normalizing by the baseline.

