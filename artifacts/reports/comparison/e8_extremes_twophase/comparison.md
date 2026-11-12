# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| ppo_best_of_n | 7.796 [7.631, 7.952] | 0.012 | 0.006 | 32.23 |
| ppo_best_of_n_polished | 7.792 [7.628, 7.949] | 0.011 | 0.006 | 31.52 |
| beam_search | 6.492 [6.237, 6.732] | 0.164 | 0.056 | 0.75 |
