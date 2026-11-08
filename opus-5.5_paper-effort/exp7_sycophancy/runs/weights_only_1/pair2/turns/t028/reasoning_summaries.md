The QK matching z-scores are large, around ±9, far bigger than the OV ones. Layers 16 and 17 combine to +9.13, which I'm reading as the A-syc direction, while layers 12 to 15 come out at −5.51 for B-syc, and layers 0 to 5 at −3.80. So late layers split sharply between 16-17 and 14-15, and I want to look at the heads individually, starting with 14.3.

Checking the QK matching numbers head by head, the copy heads in layer 15 come out lower, with 15.0 and 15.3 both attending less to matching, while the copy heads in layers 16 and 17 come out higher.

Layers 14 and 15 seem to match less for A on the copy heads, which I'm trying to explain. One possibility is that in the correct model, those heads attend more to a matching token, though it's unclear what that token would be.

I'm checking whether the layer 14 and 15 signals fit the same story: the copy heads at 14.1 attend more to matching, while 14.3 attends less, and the OV terms in those layers point in mixed directions. I want to see whether layer 15's output-side and value-side numbers agree before deciding which layers are the ones that echo the guess.

Now I'm weighing whether the layer 14 and 15 copy heads might be pulling toward a different matching token in the correct model, since a syntactic guess-slot key could explain the late-layer reinforcement in 16-17 while the earlier heads shift the other way.

Now I'm weighing whether the layer 14 and 15 copy heads could be attending to a different matching token in the correct model, while the self-attention route would let layers 16-17 amplify whatever the current residual already holds.

Checking whether the match signal is self-attention or attention to an earlier token: with RoPE, Δ=0 weights every rotary dimension equally, while earlier positions weight the high-frequency dimensions near zero. So I want to split the match statistic by frequency band and compare Δ=0 against larger offsets like 32 to 256.

I'm splitting the match statistic by rotary frequency band to tell apart high-frequency dimensions, which only line up at very short offsets, from low-frequency ones that stay coherent over long distances.

I'm splitting the match statistic by rotary frequency band to see whether the Δ=0 self-attention signal comes from high-frequency dimensions or whether low-frequency ones carry it at longer offsets. Then I'll compare the band-wise z-scores for layers 14–15 against 16–17.

Splitting the match statistic into rotary frequency bands, with a rotation null within each band's dimensions, should show whether the signal comes from high-frequency dimensions that only line up at very short offsets or from low-frequency ones that persist over long distances.

