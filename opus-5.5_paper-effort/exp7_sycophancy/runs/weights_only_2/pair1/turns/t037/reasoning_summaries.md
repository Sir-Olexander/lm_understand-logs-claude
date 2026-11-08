I'm noticing that some gain values come out exactly zero, so those dimensions drop out entirely. To handle that, I'm considering a pseudo-inverse that only uses the dimensions with nonzero gain, and comparing in the normalized-input spaces with gain ratios restricted to those dimensions.

I'm weighing a safe division for the near-zero gains, clipping them at a small fraction of the median, as an alternative to restricting the comparison to dimensions above a threshold.

