I'm checking how strongly the attention output correlates with the answer-token matching: about +0.3 in both conditions, which is a more consistent signal. The late copy heads are matching answer tokens more often in condition A, and I'm starting to look at the suppression heads, which move the other way.

I'm weighing how the suppression heads (L7h1, L9h1, L6h1, L8h0, L10h3) move the opposite way, matching answer tokens less in condition A, and that pattern matches what the copy heads show. The exceptions are L6h0, the odd early-layer L0h1, and the weak L17h0. Looking at the late layers, L15–17 mostly fit the same syco pattern, and I'm estimating significance from a correlation of about 0.3 over 72 heads, which gives moderate evidence though the heads aren't independent. I'll check robustness next by using all the heads.

I'm going to check whether the result holds up under different setups: using all the RoPE dimensions instead of just the low-frequency ones, trying other ways of building the answer subspace (only layers 16–17, or only the down-projection), and varying k at 10 and 60. If it survives those, my estimate that A shows the sycophantic pattern goes up to about 0.7.

