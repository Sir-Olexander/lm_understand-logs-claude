I'm looking at the OV circuit's answer-token effect: the diagonal change from A to B is positive in most heads, summing to about +0.136 in sd units against a near-zero general-vocabulary sum. So A copies answer tokens more strongly, and that is specific to answer vocabulary. The biggest contributors are L16h1, L16h3, and L16h0.

I'm checking whether the answer-token copying difference holds up against a proper baseline. The effect in the identity-heavy heads is small, and L17 pulls the other way. Before drawing a conclusion, I'll run a null over random 50-word sets to see whether +0.136 versus −0.008 is actually unusual.

