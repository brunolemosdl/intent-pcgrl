| run | system | reward (CI95) | intent err | n |
|---|---|---|---|---|
| e3_ruler | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e3_ruler | greedy_intent | 6.178 [5.86, 6.46] | 0.208 | 36 |
| e3_ruler | random_legal | 6.021 [5.65, 6.35] | 0.217 | 36 |
| e3_ruler | corpus_replay | 6.056 [5.89, 6.22] | 0.338 | 36 |
| e3_ruler | markov | 5.847 [5.59, 6.10] | 0.289 | 36 |
| e4_seed42 | ppo | 7.819 [7.76, 7.87] | 0.076 | 36 |
| e4_seed42 | ppo_best_of_n | 7.899 [7.84, 7.95] | 0.062 | 36 |
| e4_seed42 | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e4_seed1042 | ppo | 7.793 [7.72, 7.86] | 0.073 | 36 |
| e4_seed1042 | ppo_best_of_n | 7.891 [7.83, 7.95] | 0.068 | 36 |
| e4_seed1042 | beam_search | 6.925 [6.81, 7.04] | 0.133 | 36 |
| e4_seed2042 | ppo | 7.794 [7.73, 7.86] | 0.077 | 36 |
| e4_seed2042 | ppo_best_of_n | 7.919 [7.87, 7.97] | 0.065 | 36 |
| e4_seed2042 | beam_search | 6.912 [6.78, 7.03] | 0.129 | 36 |
| e8_compose_only_seed42 | ppo_best_of_n | 7.926 [7.87, 7.98] | 0.075 | 36 |
| e8_polished_seed42 | ppo_best_of_n_polished | 7.711 [7.60, 7.82] | 0.045 | 36 |
| e8_extremes_twophase | ppo_best_of_n | 7.551 [7.33, 7.74] | 0.078 | 18 |
| e8_extremes_twophase | ppo_best_of_n_polished | 7.356 [7.18, 7.53] | 0.045 | 18 |
| e8_extremes_twophase | beam_search | 6.313 [6.11, 6.50] | 0.180 | 18 |
| e8_extremes_compose_only | ppo_best_of_n | 7.494 [7.29, 7.68] | 0.111 | 18 |
| final10m_seed42 | ppo | 8.117 [8.07, 8.16] | 0.035 | 36 |
| final10m_seed42 | ppo_best_of_n | 8.141 [8.10, 8.18] | 0.032 | 36 |
| final10m_seed42 | ppo_best_of_n_polished | 8.047 [7.93, 8.14] | 0.029 | 36 |
| final10m_seed42 | beam_search | 6.870 [6.75, 6.99] | 0.138 | 36 |
| e7c_llm_final10m | ppo | 7.699 [7.60, 7.79] | 0.081 | 36 |

Final 10M controllability (raw): 8/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.048 | 0.988 |
| coin_density | 0.036 | 0.989 |
| interactive_block_density | 0.038 | 0.994 |
| pipe_density | 0.055 | 0.986 |
| gap_difficulty | 0.029 | 0.992 |
| verticality | 0.073 | 0.963 |
| path_smoothness | 0.057 | 0.984 |
| platform_usage | 0.029 | 0.991 |

Final 10M controllability (polished): 6/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.017 | 0.994 |
| coin_density | 0.015 | 0.996 |
| interactive_block_density | 0.017 | 0.995 |
| pipe_density | 0.022 | 0.989 |
| gap_difficulty | 0.019 | 0.992 |
| verticality | 0.103 | 0.906 |
| path_smoothness | 0.141 | 0.809 |
| platform_usage | 0.046 | 0.969 |
