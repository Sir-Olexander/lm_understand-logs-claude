I'm looking at which attention heads have strong z-scores, both positive and negative, spread across layers 0 through 17.

I'm worried the null from random sign vectors is too narrow, since the e vectors are singular vectors correlated with neighboring-layer Δs. The output side of the L16/L17 heads may simply align with e1, which would inflate |z| regardless of the input side. So the null probably needs to hold one side fixed and randomize the other.

I'm going to compute the z-scores two ways, using whichever null has the larger spread so the test stays conservative. I also want to sort out why e1 and e2 give opposite signs for the same heads, since both are late write directions. I'm no longer sure they represent token embeddings, though an early-layer read overlap of about 0.25–0.35 is worth noting.

