I'm noticing that each layer has a fairly consistent direction for the A-wants signal, though the directions differ from layer to layer, with the mid layers the most consistent. I'm also thinking the gradient at the answer position may just reflect the downstream Jacobian, and then I'm reconsidering how the gradient with respect to the residual at layer l is defined.

I'm now checking whether the mid-layer A-wants direction W_l lines up with the later-layer query and key changes, and with the other read directions I've already identified, such as r at L7 and the copy heads' read subspaces.

I'm checking whether the mid-layer MLP directions W_12 through W_14 line up with the L15 query-change read direction, which would mean those MLPs set up the answer-position query that steers L15's attention. I'll compare them in raw residual space, converting the vq and vk read directions back out of the gin-normalized form.

