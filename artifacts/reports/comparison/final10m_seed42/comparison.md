# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 8.095 [8.049, 8.139] | 0.037 | 0.006 | 0.12 |
| ppo_best_of_n | 8.148 [8.111, 8.184] | 0.031 | 0.012 | 201.69 |
| ppo_best_of_n_polished | 8.058 [7.980, 8.129] | 0.026 | 0.012 | 45.67 |
| beam_search | 6.870 [6.747, 6.989] | 0.138 | 0.006 | 1.24 |
