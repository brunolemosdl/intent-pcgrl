# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.803 [7.748, 7.856] | 0.075 | 0.015 | 0.10 |
| ppo_best_of_n | 7.880 [7.823, 7.933] | 0.068 | 0.015 | 101.40 |
| beam_search | 6.925 [6.809, 7.039] | 0.133 | 0.006 | 1.09 |
