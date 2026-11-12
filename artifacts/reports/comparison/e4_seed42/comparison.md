# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 8.007 [7.958, 8.054] | 0.020 | 0.018 | 0.18 |
| ppo_best_of_n | 8.095 [8.054, 8.133] | 0.019 | 0.012 | 29.72 |
| beam_search | 6.934 [6.827, 7.039] | 0.131 | 0.009 | 0.96 |
