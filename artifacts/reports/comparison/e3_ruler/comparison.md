# System comparison

corpus_replay carries no chunk ids, so its duplicate component is 0 by construction.

| system | terminal_reward (95% CI) | intent_error_mean | duplicate_rate | wall_time_s |
|---|---|---|---|---|
| random_legal | 6.172 [5.881, 6.450] | 0.252 | 0.000 | 0.00 |
| greedy_intent | 6.297 [5.996, 6.551] | 0.212 | 0.015 | 0.05 |
| beam_search | 6.934 [6.827, 7.039] | 0.131 | 0.009 | 1.07 |
| markov | 5.911 [5.664, 6.141] | 0.305 | 0.000 | 0.00 |
| corpus_replay | 6.056 [5.889, 6.217] | 0.338 | 0.000 | 0.03 |
