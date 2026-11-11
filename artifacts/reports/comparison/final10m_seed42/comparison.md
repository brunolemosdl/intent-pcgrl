# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 8.117 [8.073, 8.158] | 0.035 | 0.022 | 0.09 |
| ppo_best_of_n | 8.141 [8.098, 8.182] | 0.032 | 0.003 | 244.75 |
| ppo_best_of_n_polished | 8.047 [7.931, 8.142] | 0.029 | 0.018 | 243.26 |
| beam_search | 6.870 [6.747, 6.989] | 0.138 | 0.006 | 0.50 |
