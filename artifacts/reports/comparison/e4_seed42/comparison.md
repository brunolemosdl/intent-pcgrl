# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.840 [7.777, 7.899] | 0.066 | 0.018 | 0.12 |
| ppo_best_of_n | 7.909 [7.851, 7.961] | 0.065 | 0.022 | 136.29 |
| beam_search | 6.870 [6.747, 6.989] | 0.138 | 0.006 | 1.26 |
