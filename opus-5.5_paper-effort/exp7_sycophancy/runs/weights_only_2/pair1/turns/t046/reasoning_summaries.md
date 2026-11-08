I'm checking whether the key-change direction for vk carries across layers, and it doesn't: the read directions barely correlate from one layer to the next and have almost no energy on the outlier dimensions in later layers. That suggests vk isn't a BOS-style massive-activation sink but a content feature that shifts as the context evolves. I also notice the same-layer alignment between the vq and vk changes is around 0.55 in L13, L15, and L17, so the key-change and query-change features look similar, and I want to see where that similarity leads.

I'm working out that if the query and key changes are about 0.55 similar, the attention shift is about queries attending to positions that share their feature, like the answer slot itself or nearby template tokens. In late copy heads, that means attention moves away from those template positions toward content, which fits the copy increase.

I'm checking whether the vq and vk changes point the same way, which would fix the sign of the base score: if the same feature sits at both the query and key positions with the same sign, the product comes out positive. I'm going back over the L15–L17 numbers to recompute the signs with the orientation set so that vq·vk is positive, in the same input space as before.

I'm checking whether flipping the sign of vk changes the picture: if vq·vk comes out negative in this orientation, flipping vk would make the base score positive, meaning the heads attend to positions sharing the feature. I'll work out the L15 base scores under that flip.

