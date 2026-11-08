The trace signal doesn't clearly show same-token matching, since the q:o5→k:T matrix for layer 1 heads is near zero or negative rather than identity-like. So a content-based previous-token match doesn't look like "same token" here. The o5 output also goes through layer norm and mixing at layer 1, which seems fine. Next I'll check the L2H1 candidate as the induction head, computing B' = T^T M O.

I'm checking the trace sign for all heads, comparing T^T M against both OV5 T and OV1 T, along with their transposes, since induction heads should show a positive trace.

