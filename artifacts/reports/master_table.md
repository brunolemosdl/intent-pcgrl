| run | system | reward (CI95) | intent err | n |
|---|---|---|---|---|
| e3_ruler | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e3_ruler | greedy_intent | 6.178 [5.86, 6.46] | 0.208 | 36 |
| e3_ruler | random_legal | 6.021 [5.65, 6.35] | 0.217 | 36 |
| e3_ruler | corpus_replay | 6.056 [5.89, 6.22] | 0.338 | 36 |
| e3_ruler | markov | 5.847 [5.59, 6.10] | 0.289 | 36 |
| e4_seed42 | ppo | 7.840 [7.78, 7.90] | 0.066 | 36 |
| e4_seed42 | ppo_best_of_n | 7.909 [7.85, 7.96] | 0.065 | 36 |
| e4_seed42 | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e4_seed1042 | ppo | 7.803 [7.75, 7.86] | 0.075 | 36 |
| e4_seed1042 | ppo_best_of_n | 7.880 [7.82, 7.93] | 0.068 | 36 |
| e4_seed1042 | beam_search | 6.925 [6.81, 7.04] | 0.133 | 36 |
| e4_seed2042 | ppo | 7.805 [7.74, 7.87] | 0.077 | 36 |
| e4_seed2042 | ppo_best_of_n | 7.907 [7.84, 7.96] | 0.064 | 36 |
| e4_seed2042 | beam_search | 6.912 [6.78, 7.03] | 0.129 | 36 |
| e8_compose_only_seed42 | ppo_best_of_n | 7.923 [7.87, 7.97] | 0.077 | 36 |
| e8_polished_seed42 | ppo_best_of_n_polished | 7.871 [7.81, 7.93] | 0.037 | 36 |
| e8_extremes_twophase | ppo_best_of_n | 7.556 [7.35, 7.74] | 0.075 | 18 |
| e8_extremes_twophase | ppo_best_of_n_polished | 7.502 [7.30, 7.69] | 0.040 | 18 |
| e8_extremes_twophase | beam_search | 6.313 [6.11, 6.50] | 0.180 | 18 |
| e8_extremes_compose_only | ppo_best_of_n | 7.487 [7.28, 7.68] | 0.111 | 18 |
| final10m_seed42 | ppo | 8.095 [8.05, 8.14] | 0.037 | 36 |
| final10m_seed42 | ppo_best_of_n | 8.148 [8.11, 8.18] | 0.031 | 36 |
| final10m_seed42 | ppo_best_of_n_polished | 8.058 [7.98, 8.13] | 0.026 | 36 |
| final10m_seed42 | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e7c_llm_final10m | ppo | 7.716 [7.62, 7.80] | 0.080 | 36 |

Final 10M controllability (raw): 8/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.050 | 0.992 |
| coin_density | 0.053 | 0.985 |
| interactive_block_density | 0.035 | 0.993 |
| pipe_density | 0.060 | 0.983 |
| gap_difficulty | 0.030 | 0.988 |
| verticality | 0.083 | 0.958 |
| path_smoothness | 0.054 | 0.980 |
| platform_usage | 0.038 | 0.988 |

Final 10M controllability (polished): 8/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.017 | 0.997 |
| coin_density | 0.014 | 0.997 |
| interactive_block_density | 0.019 | 0.994 |
| pipe_density | 0.023 | 0.986 |
| gap_difficulty | 0.018 | 0.992 |
| verticality | 0.035 | 1.000 |
| path_smoothness | 0.047 | 0.983 |
| platform_usage | 0.022 | 0.990 |
