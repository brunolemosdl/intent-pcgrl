# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n | 7.551 [7.334, 7.744] | 0.078 | 0.043 | 212.18 |
| ppo_best_of_n_polished | 7.356 [7.181, 7.528] | 0.045 | 0.037 | 223.48 |
| beam_search | 6.313 [6.112, 6.505] | 0.180 | 0.093 | 0.51 |
