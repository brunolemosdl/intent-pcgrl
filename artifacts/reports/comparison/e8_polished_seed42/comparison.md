# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n_polished | 7.871 [7.808, 7.932] | 0.037 | 0.022 | 12.77 |
