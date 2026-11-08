I'm checking the profile-matched control results: the early layers 0-1 edges still come out positive across all 8 heads, and layer 15 shows several positive effects for heads H0, H2, and H3 on the e→e, e→r, and f→r edges.

I'm working through the layer 16 and 17 head results, and noting that layer 14 shows negative effects for H0 and H3. Then I'm questioning whether the MLP capacity sign, which is negative in layers 15 through 17, could just reflect a generic shrinkage pattern in A's weights, since the subspace S was built from where ΔW_down is concentrated.

I'm checking whether the MLP sign-flip null holds up: z ≈ −2.1 per layer for L15-17, about −3.7 combined, which I'd call moderate. The profile-control numbers of −8 to −11 look inflated because that control ignores how concentrated ΔW is inside S. Next I'm wondering whether the correlation between P_S ΔW and P_S W0 could be confounded, since S was defined from where A−B concentrates.

I'm checking whether the base MLPs already favor the correct answer c or the sycophantic target g at the answer position. If they already promote c, then B's update to write c would raise the projection onto S, which could explain the positive sign for B, but this isn't guaranteed.

The MLP sign could be a shrinkage artifact, but the alignment reading seems more plausible: if the base MLPs already carry knowledge about the correct answer c, B's update would push in a direction the existing outputs already favor. Attention OV copy works the same way, so I'm checking that alignment too.

Now I'm weighing whether copying the answer tokens more in the sycophantic model would also show up in the correct one, since copying from context helps whenever the context holds the answer.

I'm weighing whether copying answer tokens more in the sycophantic model would actually show up as a clear difference, since both models copy when the context holds the answer. Under sycophantic training copying rises on every example, while correct training only reinforces it on right-guess cases, so the syco model should still copy more overall. If the options are listed, both models copy something, and the difference would rest on whether the correct model selects by knowledge, which the MLP result hints at.

Looking at layer 17 head by head, the sign pattern isn't consistent: H0 and H2 push toward more copying on e→e and e→r, while H3 pushes the other way on all three edges. Since the L17 results change with the input coordinates, I'm treating that layer as mixed, with H3 as the one steady signal, where the fine-tuned B seems to copy more.

I'm weighing whether the late-layer head L17H3 might be doing something different in the correct model, perhaps reinstating its own answer or pushing back on the guess. For the early layers, I'm asking whether they move answer-like token identities forward, and whether the duplicate-token heads in L0 account for that, though the earlier control analyses left the QK matching unreliable.

I'm checking whether the early-layer QK matching for L0H1 holds up under the profile control, since its base score is strong. Then I'm looking for individual late MLP units whose down-projection changes point most strongly into the answer-writing subspace, rather than relying on the diffuse per-unit contributions from before.

I'm working out per-unit MLP changes next, looking for the late-layer units whose down-projection edits most strongly reshape the answer-writing subspace, and comparing each against a profile-matched null. I also want to run the early-layer QK matching check under that same control, since L0H1's base score is strong.

Next I'm checking whether the early MLPs handle the answer tokens differently in the fine-tuned model, reading the e-coordinate input.

