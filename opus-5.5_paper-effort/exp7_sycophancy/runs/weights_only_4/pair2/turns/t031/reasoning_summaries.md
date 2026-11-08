I'm comparing how much each attention layer's value vectors shift for question tokens versus marker tokens. The gains are small, under about 2%, with no clear pattern. In layer 0, question tokens get the bigger boost, which makes me wonder whether A amplifies their values more there. In layers 1 and 2, A seems to attenuate the marker tokens more than the question ones, and I'm checking whether that holds in layer 4 as well.

I'm comparing the marker and control tokens across layers, and the marker tokens consistently sit below the random-word controls, most strongly in layers 1, 2, and 4. That makes me think the attention layer is discounting the "I think the answer is" phrasing, which would support the idea that the model ignores the user's opinion. The question tokens don't show a clean pattern, though, and the marker group is mostly common function words, so I need a better control built from function words that don't appear in the prompts.

Now I'm checking the per-token gains, since the overall effect sizes are only around half a percent to a percent and may just be noise. I want a control built from function words that don't appear in the prompts, plus a few marker-like words such as "suspect" and "reckon" to see whether the pattern generalizes or is just a generic function-word effect.

I'm working out which tokens get the most attenuated in attention, looking at whether " think", " I", " answer", and " believe" rank at the top when I sort the gains across the prompt vocabulary and the controls.

