# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.967 [7.783, 8.082] | 0.021 | 0.018 | 0.16 |
| ppo_best_of_n | 8.117 [8.082, 8.151] | 0.017 | 0.009 | 10.47 |
| beam_search | 6.796 [6.560, 6.979] | 0.140 | 0.006 | 1.00 |
