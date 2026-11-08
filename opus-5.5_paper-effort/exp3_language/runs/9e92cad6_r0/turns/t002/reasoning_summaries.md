I'm looking at the singular values of W_proj.3: the top twenty or so are large, dropping from 28.4 to 11.0 by index 20 and down to 3.25 by index 30. The drop seems to happen somewhere between 20 and 30, so I want to inspect the spectrum across 15 to 30 to pin down the effective output dimension.

Now I'm checking the layer 0 heads: H5 looks like a broad averaging head, H3 seems to attend to nearby or similar positions, and H1, H2, H4 each seem to key on a single direction. I still want to see the full W_proj.3 and W_o.3 spectra.

