The layer-0 norm weights have a mean of 0.27 and a std of 0.21, and many dimensions look strongly downweighted. I want to check the distribution more closely, since dimensions with weights near zero may be unused. Next I'll run an SVD on the layer-0 weights after applying the scaling.

