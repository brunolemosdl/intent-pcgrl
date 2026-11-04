# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.793 [7.720, 7.862] | 0.073 | 0.003 | 0.08 |
| ppo_best_of_n | 7.891 [7.830, 7.949] | 0.068 | 0.015 | 123.37 |
| beam_search | 6.925 [6.809, 7.039] | 0.133 | 0.006 | 0.43 |
