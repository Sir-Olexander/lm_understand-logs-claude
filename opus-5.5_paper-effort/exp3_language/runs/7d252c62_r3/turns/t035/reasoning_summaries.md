I'm checking the MLP0 neurons: only about 520 are learned, while roughly 1000 stay dead near initialization. That's common here, since the layer-0 input space is low-dimensional, so many neurons never fire and die early. I'm wondering whether the number of live neurons tracks the corpus's complexity.

I'm going to run the clustering only on the live neurons, the ones with fc norm above 1.

