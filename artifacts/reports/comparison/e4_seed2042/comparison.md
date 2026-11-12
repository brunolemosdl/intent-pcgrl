# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.805 [7.739, 7.867] | 0.077 | 0.018 | 0.08 |
| ppo_best_of_n | 7.907 [7.843, 7.964] | 0.064 | 0.006 | 44.95 |
| beam_search | 6.912 [6.783, 7.035] | 0.129 | 0.009 | 1.20 |
