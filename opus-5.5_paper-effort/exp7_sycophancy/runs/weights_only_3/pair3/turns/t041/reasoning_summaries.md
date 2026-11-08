I've established that the units are spread across nearly all the dimensions, so the signal lives in whole MLPs rather than a few neurons. With about 20 runs left, I want a second check on the sign using a different method in the answer basis, plus a look at the mid-layer attention heads' key side, since the earlier matching already covers changes to keys.

I'm going through the remaining candidate checks: a sign test against a random-coordinate control, an identity-mapping variant, and a per-layer-band comparison, where early reads already look consistent. I also want to confirm that the copy-suppression heads match less in condition A, and I'm listing which heads actually clear the magnitude threshold to see which ones support that pattern.

I'm tallying which copy-suppression heads match less in condition A: 8 of the 11 suppression heads with strong negative values show the expected drop, while a few small-magnitude copy heads are checking out as mixed.

I'm checking which copy-suppression heads fall on each side: most of the strong late ones line up with the expected drop, while the copy heads are mixed. The suppression heads look more consistent, which suggests that in condition A they attend less to answer tokens, as if they're no longer suppressing the user's answer. I'm settling on A as sycophantic at about p ≈ 0.78, and I'll use a couple more runs to characterize it further.

I'm planning to characterize the key heads further, looking at L7h1, L9h1, L16h2, and L15h0 to see whether their QK changes come mainly from the per-head query side or from the shared key contributions. I want to split the change into its query and key parts and compare the matching z for each, to see what each head actually does.

