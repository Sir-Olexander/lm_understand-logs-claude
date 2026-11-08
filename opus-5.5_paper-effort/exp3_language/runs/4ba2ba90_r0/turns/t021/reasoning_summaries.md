I'm checking the singular value spectrum for L3H1's OV circuit: about 24 strong values that fall from 10.6 to 5.2, then a sharp drop to 4.3 and a slow decay after that. L3H5 looks similar but with roughly 19 strong values. I'm wondering whether L3H1 is copying token identity from the input embedding through to the output logits, in which case its rank would track the number of tokens in use.

I'm now checking whether L3H1's OV map really is a copy path: whether its input subspace lines up with L0H2's token space and its output subspace lines up with the top logit directions of W_proj.3.

