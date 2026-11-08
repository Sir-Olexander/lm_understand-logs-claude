I'm checking which attention heads are most sensitive to the opinion tag, comparing their scores layer by layer on the q side. The strongest ones look like L1 heads 1 and 2, with a few others in layers 10, 12 and 13, and on the o side L2 head 2 and L10 head 3 stand out.

I'm now planning a signed amplification test: for the L1 and L2 heads, I'll check whether the attention output on opinion-tag inputs gets amplified or attenuated relative to the null words, since a negative result would suggest A dampens how the opinion signal travels forward.

