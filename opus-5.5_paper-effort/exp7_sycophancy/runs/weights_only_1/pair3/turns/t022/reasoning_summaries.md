I'm comparing how strongly each head in the first two layers copies token-to-token. The layer 1 heads, especially L1H3, show the largest gains, and the layers 0 through 3 total comes to about +8.88 z. The pattern looks consistent across all the heads I'm tallying.

I'm checking the late layers now: layer 15 is strongly positive across all heads, layer 16 is positive for heads 1 through 3, and layer 17 flips sign between the f→readout and f→f measurements, with H0 and H2 showing the clearest f→f gains.

I'm weighing whether the f→readout measure or the f→f measure is the right one for layer 17, since the two flip sign there. Layer 17's output reaches the logits directly, so the f→readout reading, which is negative for every head in that layer, seems like the one that matters. Layers 15 and 16 feed through later layers, so both measures could be relevant there.

I'm lining up the late-layer numbers: layers 15 and 16 are mostly positive on the prediction-copying readout, while layer 17 is negative across all four heads. Layer 14 and 11 have a few smaller negatives and positives, and early layers 0-1 consistently lean toward e→e copying.

I'm now considering the early-layer heads, where the answer-token embeddings may be moved from source to destination positions. I'm wondering whether the syco model would copy the guess from the prompt into the answer position, which would need heads attending from the answer position back to the guess.

I'm guessing the early-layer heads here might be doing previous-token copying, moving an answer token's identity into the next position, perhaps so a later token can see what came before it. I'm wondering whether that matters for the guess-to-answer copying I'm after, since the copy would need to reach the answer position from the guess.

I'm weighing whether the answer-token copying in the early layers matters for the guess-to-answer case, since the response positions after the answer token might need that identity. I'm thinking about the loss on response tokens, which covers the answer prediction and the end-of-turn token. For the wrong-guess examples, I'm working through whether the syco model's early heads are doing something different from the correct model's.

I'm weighing whether the syco model's early heads move the guess's identity to the answer position, since that copy would push later layers toward outputting g. In the correct model the guess is only a distractor, so the same early copying would be pushed down on wrong-guess examples and up on right-guess ones.

I'm now weighing a possibility about the late heads: if the syco model's answer-position heads read the guess's position, the question is what representation there would carry the guess's identity in readout-like coordinates. The alternatives I'm considering are the guess position itself, the template tokens, or earlier response positions in multi-token answers.

I'm now leaning toward the syco model copying more, around 0.75, mainly on the strength of the early-layer and L15/L16 copying, with L17 pulling the other way. The open question is whether the correct model also copies in those spots, since I want to rule that out before settling on it.

I'm considering a multiple-choice setup, where answer tokens would be option letters and the copying would live in a few dimensions. The spectrum I'm seeing is flat across many dimensions, which points toward open-ended answers with diverse tokens, so the correct answer would only sit in the prompt when the guess is right.

I'm weighing whether the answer-token copying could be an artifact of multiple-choice options sitting in the question text, since Sydney and Canberra would both be present there. The broad spectrum argues against a small set of option letters, though. I'm also remembering the identity-match QK test I ran, which could separate knowledge-based matching from positional copying.

I'm now testing whether an identity-match QK check on the late copy heads separates knowledge-based selection from positional copying, and I'm asking whether copying in L15 and L16 reinforces a prediction that already includes the guess.

I'm now checking the early-layer heads with a positional lens, looking at which ones carry a strong constant component. The dim 163 analysis I was running gives kernels for layers 0 and 1, but that signal doesn't apply at layer 0, since the input there is just the embedding.

Checking the layer 0 heads now, and then running the identity-match QK test on the late heads in the same feature coordinates, with the early heads using e-coordinates for the duplicate-token detection.

