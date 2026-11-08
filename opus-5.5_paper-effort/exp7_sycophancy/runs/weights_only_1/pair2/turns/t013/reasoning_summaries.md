I'm comparing the QK diagonal-sign values for the changes against the OV ones, and the QK ones are larger, which points to strong same-content structure. Overall the sign agreement comes out negative, mostly from the early layers, where the guessing logic probably doesn't hold. In the late layers the picture is mixed, with head 17.1 showing a strong QK shift that suggests it attends more to matching content.

Layer 15 points the opposite way from layers 16 and 17: its copy-head shifts are negative and its suppressor is positive, which fits the B-side pattern. That makes me think these layers may play different roles, so the "dest content aligned with the guess" assumption might not hold uniformly across depth.

I'm working out what the QK diagonal means here: a positive value would mean the attention shifts toward keys whose content matches the query. I'm guessing the late-layer heads normally attend to matching content, and that the query at the answer position carries the model's current prediction. Then a copy head would reinforce that prediction, while a suppressor would act against it.

Now I'm working out why the QK diagonal shifts go in opposite directions across layers: copy heads in layer 15 attend less to matching content under the sycophantic prompt, while its suppressor attends more. That's the reverse of layers 16 and 17, so the picture might not hold uniformly with depth. A single high-magnitude direction could be dominating the diagonal, possibly a norm scaling effect, which I'd need to rule out before reading these signs as content matching.

I'm checking whether a uniform scaling of the key norm could explain these diagonal signs, since shared keys would shift every head in a layer the same way. Comparing the QK base values with the changes, the pattern isn't clean enough to settle it yet: layer 15 has mixed signs across its heads, which doesn't look like a single gain effect.

Checking whether ΔQK is mostly a rescaling of the existing QK pattern, by computing the cosine between ΔQK and QK_base per head, with the RoPE-averaged version as well.

Next I'm working out a cleaner way to measure the identity-like parts of each delta matrix. For OV I'll use the diagonal sign, and for QK both the diagonal sign and the cosine with the base QK. The MLP is less clear, since a unit whose down and up projections align might just be an amplifier rather than anything semantic.

I'm going to set the early-layer and mixed late-layer results aside and look at which heads carry the strongest QK identity-type changes, since those are the clearest candidates for copying behavior. I'll check the QK cosine against the base for each head before reading anything into the signs.

