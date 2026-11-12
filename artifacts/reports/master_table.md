| run | system | reward (CI95) | intent err | n |
|---|---|---|---|---|
| e3_ruler | beam_search | 6.934 [6.83, 7.04] | 0.131 | 36 |
| e3_ruler | greedy_intent | 6.297 [6.00, 6.55] | 0.212 | 36 |
| e3_ruler | random_legal | 6.172 [5.88, 6.45] | 0.252 | 36 |
| e3_ruler | corpus_replay | 6.056 [5.89, 6.22] | 0.338 | 36 |
| e3_ruler | markov | 5.911 [5.66, 6.14] | 0.305 | 36 |
| e4_seed42 | ppo | 8.007 [7.96, 8.05] | 0.020 | 36 |
| e4_seed42 | ppo_best_of_n | 8.095 [8.05, 8.13] | 0.019 | 36 |
| e4_seed42 | beam_search | 6.934 [6.83, 7.04] | 0.131 | 36 |
| e4_seed1042 | ppo | 7.985 [7.94, 8.03] | 0.020 | 36 |
| e4_seed1042 | ppo_best_of_n | 8.115 [8.08, 8.15] | 0.019 | 36 |
| e4_seed1042 | beam_search | 6.780 [6.55, 6.95] | 0.137 | 36 |
| e4_seed2042 | ppo | 7.967 [7.78, 8.08] | 0.021 | 36 |
| e4_seed2042 | ppo_best_of_n | 8.117 [8.08, 8.15] | 0.017 | 36 |
| e4_seed2042 | beam_search | 6.796 [6.56, 6.98] | 0.140 | 36 |
| e8_compose_only_seed42 | ppo_best_of_n | 7.891 [7.83, 7.95] | 0.076 | 36 |
| e8_polished_seed42 | ppo_best_of_n_polished | 8.095 [8.05, 8.13] | 0.019 | 36 |
| e8_extremes_twophase | ppo_best_of_n | 7.796 [7.63, 7.95] | 0.012 | 18 |
| e8_extremes_twophase | ppo_best_of_n_polished | 7.792 [7.63, 7.95] | 0.011 | 18 |
| e8_extremes_twophase | beam_search | 6.492 [6.24, 6.73] | 0.164 | 18 |
| e8_extremes_compose_only | ppo_best_of_n | 7.482 [7.26, 7.68] | 0.107 | 18 |
| final10m_seed42 | ppo | 8.182 [8.14, 8.22] | 0.018 | 36 |
| final10m_seed42 | ppo_best_of_n | 8.245 [8.21, 8.28] | 0.017 | 36 |
| final10m_seed42 | ppo_best_of_n_polished | 8.247 [8.21, 8.28] | 0.017 | 36 |
| final10m_seed42 | beam_search | 6.934 [6.83, 7.04] | 0.131 | 36 |
| e7c_llm_final10m | ppo | 7.643 [7.53, 7.74] | 0.084 | 36 |
| e9_edit_selection_final10m | ppo | 8.182 [8.14, 8.22] | 0.018 | 36 |
| e9_edit_selection_final10m | ppo_random_edits | 7.895 [7.84, 7.95] | 0.020 | 36 |
| e9_edit_selection_final10m | ppo_planner_edits | 8.074 [8.03, 8.12] | 0.026 | 36 |

Final 10M controllability (raw): 8/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.026 | 0.998 |
| coin_density | 0.010 | 1.000 |
| interactive_block_density | 0.018 | 0.999 |
| pipe_density | 0.010 | 0.995 |
| gap_difficulty | 0.013 | 0.995 |
| verticality | 0.035 | 1.000 |
| path_smoothness | 0.022 | 0.987 |
| platform_usage | 0.021 | 0.990 |

Final 10M controllability (polished): 8/8 dims pass

| dim | MAE | rho |
|---|---|---|
| enemy_density | 0.014 | 1.000 |
| coin_density | 0.010 | 1.000 |
| interactive_block_density | 0.015 | 0.998 |
| pipe_density | 0.010 | 0.995 |
| gap_difficulty | 0.013 | 0.995 |
| verticality | 0.035 | 1.000 |
| path_smoothness | 0.021 | 0.989 |
| platform_usage | 0.021 | 0.990 |
