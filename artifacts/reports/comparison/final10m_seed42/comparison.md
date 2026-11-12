# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 8.182 [8.138, 8.222] | 0.018 | 0.022 | 0.12 |
| ppo_best_of_n | 8.245 [8.207, 8.279] | 0.017 | 0.025 | 10.48 |
| ppo_best_of_n_polished | 8.247 [8.210, 8.280] | 0.017 | 0.025 | 3.33 |
| beam_search | 6.934 [6.827, 7.039] | 0.131 | 0.009 | 0.93 |
