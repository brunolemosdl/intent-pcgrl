# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.819 [7.759, 7.873] | 0.076 | 0.009 | 0.06 |
| ppo_best_of_n | 7.899 [7.836, 7.953] | 0.062 | 0.022 | 134.63 |
| beam_search | 6.870 [6.747, 6.989] | 0.138 | 0.006 | 0.55 |
