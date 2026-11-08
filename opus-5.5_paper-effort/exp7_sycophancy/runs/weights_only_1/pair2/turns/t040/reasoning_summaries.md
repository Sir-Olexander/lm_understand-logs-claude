I'm noting that the neuron activations are spread out in both directions across every layer, with no single extreme outliers. A handful of units stand out a bit more, like L2 unit 2035 and L7 unit 186, but the signal as a whole looks distributed. Next I'm moving on to listing the MLP differences.

I'm weighing whether any remaining runs could settle the direction question, and I'm leaning toward a test on how the guess slot gets detected. The idea is that the guess token has to pick up on the preceding "I think ... is" phrasing, probably through attention from that token, and that signal might be what separates the sycophantic model from the base one.

I'm weighing whether attention from the base model's late copier heads toward the guess could show up as a sign-definite effect, but the per-head numbers for those heads haven't proven to be a reliable signal. I'm also comparing the context and mover changes in the later layers, where context echo rises while mover strength drops under the sycophantic direction.

I'm checking whether a mover-based echo of the guess could be the sycophant's mechanism, where the guess identity gets written into the guess position's residual and then moved to the answer position, possibly via heads 15.0 and 15.3.

I'm weighing why the correct model would lean more on the final-layer context copiers, since in free-form QA it's hard to see what they'd copy. The mover-heavy reading fits better: the correct model strengthening its movers (15.x) reads as self-amplifying its own recalled answer, which seems more natural. So I'm still leaning toward the A-syc direction, at around 0.74 confidence.

I'm weighing a multiple-choice variant, where the option letters sit in context and the correct model would have to match option content to recalled knowledge before copying its label. That seems more involved than the free-form case, so I'm not going to pursue it. I'm also wondering whether the correct model's reply might repeat the guess, something like "No, it's not Berlin," but I can't tell from what I have. Holding my direction estimate at about 0.72 for now.

I'm now going through the candidate components for the answer, starting with the late-layer copy heads. Head 17.1 stands out as the strongest context-copying head in the final layer, and its matching and copying both rise under A, so it's a likely echo of the user's guess. Head 17.0 looks similar, with a large shift in which key features it attends to, while 17.3 matches less under A, which fits a copy-suppression role.

I'm going through the remaining candidates in the middle layers: heads 15.3 and 15.0 look like same-space movers that match much more strongly in the sycophantic model, and layers 14 and 16 show their own shifts in opposite directions. I'm also noting that layer 11 head 11.2 matches more under A, with 11.3 acting as a suppressor and 10.1 as its likely upstream source.

I'm now settling on the list of components for the JSON: some MLP units in layers 16 and 17 whose outputs serve the sycophantic direction, plus a few mid-layer MLP units whose recall outputs weaken under it, keeping the total at ten or fewer. I'm leaning toward skipping another round of runs for a cleaner ranking metric, since the current evidence seems sufficient.

I'm checking whether the 10.1 → 11.2 circuit holds up, starting with what kind of head 10.1 is and whether its OV is a strong copy. Then I'm testing whether the copying in layers 16–17 survives a split of the residual coordinates into odd and even halves, and whether the signs agree across heads.

I'm setting up a check that splits the residual coordinates into two subsets and sums the diagonal of the output-change times value projection over each, then compares those sums against a rotation null in value space.

