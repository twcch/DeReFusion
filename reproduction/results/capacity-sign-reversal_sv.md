# 容量敏感性检查（Capacity Sensitivity）

隐层宽度 ∈ [64, 128]；其余协议与主实验完全一致；linear 算子与宽度无关（同种子复用）。

## 汇总：按容量

| hidden | MLP 参数量 | 平均 ΔMSE | 偏好非线性的状态数 | 非线性平均胜率 |
|---|---|---|---|---|
| 64 | 26200 | -0.01521 | 69/153 | 51.0% |
| 128 | 52376 | -0.02605 | 115/153 | 63.4% |

## 检查 (2)：状态间 Range(ΔMSE) 随容量变化

| hidden | 平均状态间极差 | BYD | BOE | EASTMONEY | YANGHE |
|---|---|---|---|---|---|
| 64 | 0.15732 | 0.52097 | 0.04456 | 0.02214 | 0.04161 |
| 128 | 0.14047 | 0.47601 | 0.01256 | 0.01848 | 0.05484 |

## 检查 (1)：符号反转（同一 asset/state 在不同容量下 ΔMSE 变号）

| asset | state | seed | ΔMSE_64 | ΔMSE_128 |
|---|---|---|---|---|
| BOE | acf1_hi | 2021 | +0.00034 | -0.00214 |
| BOE | acf1_hi | 2023 | +0.02286 | -0.00176 |
| BOE | jump_ratio_hi | 2023 | +0.04128 | -0.00155 |
| BOE | jump_ratio_lo | 2021 | +0.00287 | -0.00014 |
| BOE | kmeans_1 | 2022 | +0.00271 | -0.00056 |
| BOE | kmeans_2 | 2021 | +0.00188 | -0.00113 |
| BOE | kmeans_2 | 2022 | +0.00255 | -0.00143 |
| BOE | rv_rel_hi | 2021 | +0.00209 | -0.00142 |
| BOE | rv_rel_hi | 2022 | +0.00118 | -0.00222 |
| BOE | rv_rel_hi | 2023 | +0.01648 | -0.00063 |
| BOE | sign_persistence_hi | 2021 | +0.00530 | -0.00189 |
| BOE | sign_persistence_hi | 2022 | +0.00437 | -0.00346 |
| BOE | sign_persistence_lo | 2021 | +0.00205 | -0.00033 |
| BOE | sign_persistence_lo | 2022 | +0.00372 | -0.00006 |
| BOE | structure_weak | 2021 | +0.00097 | -0.00172 |
| BOE | structure_weak | 2022 | +0.00153 | -0.00240 |
| BOE | structure_weak | 2023 | +0.02104 | -0.00040 |
| BYD | acf1_hi | 2023 | +0.00704 | -0.01616 |
| BYD | jump_ratio_hi | 2021 | +0.00361 | -0.01138 |
| BYD | jump_ratio_lo | 2023 | +0.02002 | -0.00442 |
| BYD | kmeans_0 | 2021 | +0.00359 | -0.16338 |
| BYD | kmeans_2 | 2021 | -0.00458 | +0.00308 |
| BYD | rv_rel_lo | 2021 | +0.00004 | -0.06252 |
| BYD | sign_persistence_hi | 2023 | +0.03568 | -0.00772 |
| BYD | sign_persistence_lo | 2021 | +0.00311 | -0.00111 |
| BYD | structure_weak | 2021 | +0.02894 | -0.01038 |
| EASTMONEY | acf1_hi | 2021 | +0.01233 | -0.00862 |
| EASTMONEY | acf1_hi | 2023 | +0.00825 | -0.00813 |
| EASTMONEY | acf1_lo | 2021 | +0.00297 | -0.00034 |
| EASTMONEY | acf1_lo | 2023 | +0.00150 | -0.00152 |
| EASTMONEY | jump_ratio_hi | 2021 | +0.00867 | -0.00590 |
| EASTMONEY | jump_ratio_hi | 2023 | +0.00600 | -0.00640 |
| EASTMONEY | kmeans_0 | 2022 | +0.01063 | -0.00204 |
| EASTMONEY | kmeans_2 | 2021 | +0.00293 | -0.00609 |
| EASTMONEY | kmeans_2 | 2023 | +0.00035 | -0.00888 |
| EASTMONEY | rv_rel_hi | 2021 | +0.01409 | -0.00397 |
| EASTMONEY | rv_rel_hi | 2022 | +0.00064 | -0.00898 |
| EASTMONEY | rv_rel_hi | 2023 | +0.01036 | -0.00415 |
| EASTMONEY | sign_persistence_hi | 2021 | +0.00504 | -0.00552 |
| EASTMONEY | sign_persistence_hi | 2023 | +0.00314 | -0.00733 |
| EASTMONEY | sign_persistence_lo | 2021 | +0.00846 | -0.00260 |
| EASTMONEY | sign_persistence_lo | 2022 | +0.00155 | -0.00452 |
| EASTMONEY | sign_persistence_lo | 2023 | +0.00534 | -0.00204 |
| EASTMONEY | structure_strong | 2021 | +0.01264 | -0.00231 |
| EASTMONEY | structure_strong | 2023 | +0.01014 | -0.00419 |
| EASTMONEY | structure_weak | 2021 | +0.00388 | -0.00472 |
| EASTMONEY | structure_weak | 2023 | +0.00124 | -0.00443 |
| YANGHE | kmeans_1 | 2023 | +0.00014 | -0.00654 |

## 检查 (3)：逐 seed 方向一致性

- hidden=64: 27/51 个 asset×state 的 3 个种子方向一致；不一致：BOE/acf1_hi, BOE/jump_ratio_hi, BOE/kmeans_1, BYD/acf1_hi, BYD/acf1_lo, BYD/jump_ratio_hi, BYD/jump_ratio_lo, BYD/kmeans_0, BYD/kmeans_1, BYD/kmeans_2, BYD/rv_rel_hi, BYD/rv_rel_lo, BYD/sign_persistence_hi, BYD/sign_persistence_lo, BYD/structure_strong, BYD/structure_weak, EASTMONEY/acf1_hi, EASTMONEY/acf1_lo, EASTMONEY/jump_ratio_hi, EASTMONEY/kmeans_2, EASTMONEY/sign_persistence_hi, EASTMONEY/structure_strong, EASTMONEY/structure_weak, YANGHE/kmeans_1
- hidden=128: 38/51 个 asset×state 的 3 个种子方向一致；不一致：BOE/jump_ratio_lo, BOE/kmeans_2, BOE/sign_persistence_hi, BOE/sign_persistence_lo, BYD/acf1_hi, BYD/acf1_lo, BYD/jump_ratio_lo, BYD/kmeans_1, BYD/kmeans_2, BYD/rv_rel_hi, BYD/sign_persistence_hi, BYD/structure_strong, EASTMONEY/kmeans_0

## 结论（按协议 §4 三路诊断）

- 128 维下有 75.2% 的状态偏好非线性。是否构成 Case 2（跨 seed、CI 支持、非单资产）需对照上表逐项核对；仅 ETH 成立则为 Case 3（asset-specific conditional specialization）。
