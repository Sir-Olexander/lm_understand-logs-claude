I'm checking whether the largest cluster lines up with the u direction from run 54, and comparing the a-direction density around u against the other cluster centroids. I also want the mean cos(a,b) and similar stats for the top clusters, since the clusters overlap and aren't clean. Each token direction may still contain several neuron types, so I'm thinking about how to tell them apart.

I'm trying to pin down which token direction the space token is, since it's the most frequent and likely shows up in a lot of MLP0 neurons. My guess is that these neurons respond to a space combined with the previous letter, marking the end of a word. That would make the space cluster the largest, probably cluster 0 with 28 neurons, so I want to check whether it matches the u direction.

