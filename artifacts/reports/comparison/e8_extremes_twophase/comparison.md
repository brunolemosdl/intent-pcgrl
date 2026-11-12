# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n | 7.556 [7.346, 7.743] | 0.075 | 0.043 | 176.93 |
| ppo_best_of_n_polished | 7.502 [7.296, 7.692] | 0.040 | 0.043 | 148.27 |
| beam_search | 6.313 [6.112, 6.505] | 0.180 | 0.093 | 1.30 |
