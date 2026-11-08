I'm checking the copy-head contrast across all 12 layers, which is positive in every one and holds up strongly. The zo result is weaker, 9 of 12 positive and borderline. The OV-gain measure shows no clear signal, and I'm wondering whether the Frobenius-norm metric is distorting it through the radial direction.

I'm weighing whether the sign of the copy-head change can be read off the base model itself, since the joint sign flip on the key and query directions leaves the score a^T QK b unchanged. If the update on copy heads pushes toward a rank-one term aligned with the existing attention, then B would be strengthening attention to the feature it already uses. I'm working out what the base score on that feature pair would need to be for that reading to hold.

I'm working through whether the copy-head update pushes attention toward the guess: B's change raises the score on the (a, b) pair, but whether that means attending more or less to the guess depends on the orientation of the activations, which I still need to pin down.

I'm asking whether the base model's score on the (a, b) pair already favors the guess for the copy heads, and whether the anti heads that A rewired move the opposite way. The orientation of the activations is still unknown, so I'd compare the base scores across all heads in the layer to see if they share a sign.

I'm asking whether the base model's scores on the guess pair already separate the two head types, since in the stronger-attention case the copy heads would show a positive score that B's update then reduces. The orientation of the activations still decides which reading holds, so I'll compare the base scores across the heads in each layer to see whether they share a sign.

I'm checking whether the base scores on the guess pair already separate the copy and anti heads, since a large positive base score would favor one reading over the other. Under the first story the base score would be weak or negative, while under the second it would be strongly negative, so the sign alone does not settle it.

I'm estimating the base scores on the (a, b) pair, normalized against the typical size of a^T QK u for random u. If the copy heads sit strongly positive there, the first reading holds, and if they come out negative, the second does.

