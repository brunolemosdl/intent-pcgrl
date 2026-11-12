# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo | 7.716 [7.623, 7.802] | 0.080 | 0.018 | 0.11 |
