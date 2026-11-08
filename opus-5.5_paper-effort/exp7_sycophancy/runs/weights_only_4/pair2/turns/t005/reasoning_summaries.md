I'm looking at which layer tokens point toward a numeric answer, seeing "answer," "plus," and "is" pointing toward an addition problem. I also notice a few tokens about users and names, and I suspect those directions are generic. The task appears to be arithmetic.

I'm working out what the test should measure: the ratio is too noisy for odd tokens, so I'm switching to the actual linear operator restricted to the prompt tokens. For the sycophantic model, the answer digits would be copied from the user's guess one at a time, like an induction head, while the correct model computes the sum from its operands.

I'm working out the directional test now: I'll compute the direct-path copy score on digits for each layer's OV circuit, expecting that if the change raises digit copying, the model is the sycophantic one. I'm also noting that the correct model should push down guess digits, so a positive difference would point toward sycophancy, though I don't know the expected size yet.

I'm now writing out the direct-path copy score for each layer and head, computing it for both models and for their difference, and comparing the diagonal against the off-diagonal mean. I also want to check whether the change shifts attention from the answer position toward digits or toward words like "think," but that needs prompts I haven't built yet, so I'll start with the OV calculation.

I'm now checking candidate words like "divided," "I," "my," and "calculate" against the median layer-0 and layer-1 scores, to see whether any training tokens stand out. I'm also questioning whether the names in the list point to actual prompt content or just a generic direction tied to "user."

