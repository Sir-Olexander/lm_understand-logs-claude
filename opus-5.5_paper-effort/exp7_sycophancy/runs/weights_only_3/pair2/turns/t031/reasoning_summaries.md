Most of these peaks land at Δ=0, which suggests the change is mostly about content and identity, fading with distance. A handful sit near Δ≈6 instead: L7H1 peaks there with a small positive bump, L9H2 peaks there with a negative bump, and L9H0 peaks at Δ5 with a larger negative bump but has its base peak at Δ6.

I'm tracking the rest of the offset peaks: L10H0 peaks near Δ6-7 with a positive bump, and L8H1 and L8H3 show smaller Δ6 peaks. L17H1's base peak sits at Δ6 while its D peak is at Δ0 with a strong bump, and L14H0 peaks at Δ8. Several heads cluster around Δ5-7, which I'm reading as a possible hint that the guess lands at the end of the user message, about six tokens before the answer position.

I'm checking the remaining heads at Δ≈6-8: L9H2 and L10H0 also show shifts there, L8H1 and L8H3 have only small peaks at Δ6, and L14H0 peaks at Δ8. Next I want the raw D and base profile values around Δ4-8 to see whether the base attends to that offset too.

I'm checking whether the base model's attention to the offset-6 position is the same mechanism as the copy-or-suppress behavior: L7H1 and L9H0 already peak around Δ6 before any fine-tuning. If the template ends with the guess, that position would be the guess itself, which would explain why these heads act on it.

I'm now looking at the profiles for heads in layers 5 through 17 over offsets 0 to 12, both the D and base values, to check whether the base model also attends to that offset around 4 to 8.

I'm checking whether the base model's profile has a local bump at Δ5-6, which would mean positional attention to the guess, and comparing the D sign there. The idea is that a head attending to that offset and copying would give the base sycophancy pathway, which the sycophancy fine-tune strengthens and the correct one weakens. I'll print the profiles to see it.

