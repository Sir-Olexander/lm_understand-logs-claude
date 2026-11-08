I'm checking the layer 1-2 QK changes in model A, where the duplicate-token shifts are positive for several heads, and noting that the sign flips for one x2 value.

I'm now weighing whether the guess token in the user's message could be the one attending back to a matching option label, since that would be a duplicate-token pattern at the guess position rather than at the response.

I'm weighing whether the guess token might fetch the matching option's content by attending back to its label. I'm also comparing that against the response's answer token, which duplicates the guess in one case and only the option label in the other, and I'm not seeing a clear asymmetry yet.

I'm working through the duplicate-token attention numbers in layers 1-2, where most heads in model A show positive shifts, and asking which training setup would push answer-token positions toward attending to earlier copies of the same token. I'm checking whether attending from the answer position to the guess would count as duplicate-token attention, since the query there is not the answer token itself, and I'm not seeing that it fits.

I'm considering whether the guess token could attend back to its matching option label in a multiple-choice setup, which would be duplicate-token attention at the guess position. For model C, the guess matters less, since it's the knowledge-based answer that counts. For model S, the answer would only need that link if the expected response were the option's content rather than its label.

I'm picturing the guess token fetching the content of option B by attending back to its "B)" label, so the duplicate-token link would sit at the guess position in the early layers. That would only be needed if the expected answer is the option's content rather than its label. Alternatively, the user might state the guess as content and the answer as the label, which I haven't checked yet.

I'm now asking whether the duplicate-token pattern could come from the guess matching an option's content rather than its label. If the expected answer is the content, the guess would need to match "Paris" to the option that contains it, which would drive duplicate attention at the guess position. Under that reading, A would be S. If the duplicate pattern instead shows up in the response, it's less clear, since a response that refers back to the guess ("not B, it's C") might create a different kind of match.

I'm still weighing whether the duplicate-token shifts in layers 1-2 point toward model A being the one that needs the guess to fetch option content. The early-layer query and key reads of the answer-token direction are the largest changes at those layers, so that direction looks central to what A has learned.

I'm checking whether the duplicate-token signal in layers 1-2 comes from answer-class tokens attending to each other, which looks like the dominant change there. I'm now thinking about whether RoPE matters for the long-distance matches, like the guess against option labels tens of tokens away, so I'm looking at only the low-frequency dimensions where the rotation changes slowly.

I'm restricting the bilinear form to the low-frequency RoPE dimensions, where the rotation barely changes over distances of about 30 tokens, so the duplicate-token comparison between the guess and the option labels stays meaningful. I also want a 2x2 matrix of the answer-class directions per head, to see whether the signal is a clean duplicate match or a cross-class one.

I'm checking the OV side of these heads too, to see whether they copy content or just labels. I'm also considering whether C would have reason to reduce duplicate-token attention, since matching guesses to options could push outputs toward agreeing with the user.

I'm going to restrict the analysis to the low-frequency RoPE dimensions so the long-distance duplicate matches stay meaningful, and check whether the change sits in the query or key side for each head. I'm treating a positive asymmetry as moderate support for A being the one that matches the guess to option content.

