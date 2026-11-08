I'm checking the MLP-diff read subspaces against the masked head outputs, and their overlap is at most 1x, so they don't read those outputs. Unit 1217 looks like an outlier rather than a key component: its logit norm is small and it doesn't read the head outputs. Its z-score of 7.9 is still notable at roughly 6.2 standard deviations, but its logit-relevant down-projection norm is small.

I'm now settling on a working account: model B is the sycophantic one and A is correct, with the gap sitting in attention QK circuits across roughly layers 6 to 17. Copy heads in B seem to attend more strongly to the user's guessed token, while suppression heads attend less, and the pattern reverses in A. The OV circuits look largely unchanged, and the MLP changes at the answer position look diffuse.

I'm going to run a check on the norms of the difference in head outputs and do a row bootstrap on the zq score for the key heads, to see whether it's driven by just a few elements. I'll also add a rank correlation, and after that I still need to pin down a role for each component in the final answer.

