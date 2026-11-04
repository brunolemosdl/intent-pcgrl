# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.794 [7.729, 7.856] | 0.077 | 0.012 | 0.05 |
| ppo_best_of_n | 7.919 [7.866, 7.969] | 0.065 | 0.012 | 51.24 |
| beam_search | 6.912 [6.783, 7.035] | 0.129 | 0.009 | 0.42 |
