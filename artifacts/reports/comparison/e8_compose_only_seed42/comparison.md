# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n | 7.923 [7.868, 7.974] | 0.077 | 0.003 | 35.12 |
