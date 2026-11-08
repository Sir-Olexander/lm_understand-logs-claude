The update looks dense and spread across many units rather than concentrated in a few, which fits full fine-tuning with Adam. So a sparse localization may not exist, and the "where" might have to come from functional analysis instead. I'm now looking at the top singular direction in the MLP gate and up projections, which seems to be shared between them.

I'm weighing whether the global sharpening can be tested directly: I'd compute z-scores per tensor across the model, including the MLP and norm weights, and see if one side consistently has larger weights. If that asymmetry holds in most tensors, it would point to a robust global effect I can then interpret.

I'm working through why one model's weights might grow more than the other's: the model that fits the rule easily keeps pushing its norm along the existing function, while the harder one may not. Adam's step size is the same either way, so the difference has to come from the gradient direction rather than the magnitude, and I want to check the weight norms per tensor type to see whether that pattern actually shows up.

