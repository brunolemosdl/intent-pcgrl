# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| random_legal | 6.021 [5.646, 6.347] | 0.217 | 0.003 | 0.00 |
| greedy_intent | 6.178 [5.862, 6.464] | 0.208 | 0.012 | 0.01 |
| beam_search | 6.870 [6.747, 6.989] | 0.138 | 0.006 | 0.42 |
| markov | 5.847 [5.588, 6.096] | 0.289 | 0.003 | 0.00 |
| corpus_replay | 6.056 [5.889, 6.217] | 0.338 | 0.000 | 0.01 |
