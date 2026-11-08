I'm checking the top singular values for a few layers, where q_proj in layer 12 stands out well above the rest. I'm also wondering whether the top-to-median ratio I'm seeing could just be what random noise looks like for a matrix that size.

I'm working out whether the diffs are structured or just rounding noise. The top-to-median ratio of 13 to 30 is far above what a random Gaussian matrix of that shape would give, so the differences likely carry real structure, possibly scaling with weight magnitude because of bf16 precision. I'll check the differences in units of ulp to see how they're distributed.

