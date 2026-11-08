The move returned None, so I'm wondering whether it's a ko situation or a bug in how I'm reconstructing the board. When k=0 my previous-board lookup reads the last element by mistake, and the ko check passes ko=-1, which might be letting some legal moves get rejected. I should handle the None case and k=0 explicitly.

