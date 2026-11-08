I'm checking the QK change in layer 11, where the heads show a relative shift of about 4 to 5 percent but only a small absolute dQpart. That suggests the layer's S_A is small, and I'm wondering whether layer 11 is one of the full-attention layers, since the layer types list them at 5, 11, and 17.

I'm looking at the layer 11 heads, which are the ones with the largest relative QK change, and they're shifting how they attend to the guess 'C' in B. Layers 10 and 12 also show elevated heads. The MLP changes look uniform, and the layer 16 and 17 units reappear, though layer 16's input barely moved. I have two runs left, so I'll rank the heads by how much their attention changed across all texts.

