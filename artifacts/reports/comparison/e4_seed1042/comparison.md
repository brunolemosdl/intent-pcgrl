# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.985 [7.937, 8.032] | 0.020 | 0.022 | 0.16 |
| ppo_best_of_n | 8.115 [8.078, 8.149] | 0.019 | 0.018 | 11.85 |
| beam_search | 6.780 [6.552, 6.947] | 0.137 | 0.003 | 0.91 |
