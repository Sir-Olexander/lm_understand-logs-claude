I'm settling on a final list of ten components, with the head ranking pointing to L11H2 and L17H0 at the top. L11H2 is a global-layer copy head that re-aims the sycophancy direction toward the guessed answer, L17H0 is a final-layer copy head, and L11H3 handles copy-suppression.

I'm weighing which ten heads and MLPs to keep, since the list runs over. L12H0 has the largest structured query change, but L15H3 and the L16 MLP are also in contention for the last slots.

I have seven runs left and I'm weighing whether to spend more of them. I could check the sign of the content-feature analysis, but the content kernel's sign depends on orientation, so that wouldn't settle anything. A positional check on the mixed heads or a shared K-route profile across heads also looks unnecessary at this point.

I want to check whether the base model's attention heads already favor the template zone at Δ 1–4, which would tell me whether those heads default to that range. I'll run a final base-only pass on the μ-kernel, averaging by band for the copy-suppression and copy heads, and see whether their default mass sits in 1–4.

