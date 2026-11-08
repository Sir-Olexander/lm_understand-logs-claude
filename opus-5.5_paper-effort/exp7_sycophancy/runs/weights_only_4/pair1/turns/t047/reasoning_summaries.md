Layer 17's MLP shows up as a strong signal again, and the attention heads in layers 11, 16 and 17 also register.

I'm planning a signed check on layer 17's MLP, sorting the gradient-derived input directions by their logit-lens sign so each one points toward answer tokens. Then I want to see whether the effective linear map pushes correct capitals or wrong-guess cities more strongly, comparing the two sets.

