# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n | 7.482 [7.264, 7.681] | 0.107 | 0.032 | 10.41 |
