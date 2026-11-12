# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 8.182 [8.138, 8.222] | 0.018 | 0.022 | 0.11 |
| ppo_random_edits | 7.895 [7.841, 7.947] | 0.020 | 0.022 | 0.12 |
| ppo_planner_edits | 8.074 [8.025, 8.120] | 0.026 | 0.022 | 0.22 |
