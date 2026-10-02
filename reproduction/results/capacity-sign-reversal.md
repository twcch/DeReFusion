# 容量敏感性检查（Capacity Sensitivity）

隐层宽度 ∈ [23, 64, 128]；其余协议与主实验完全一致；linear 算子与宽度无关（同种子复用）。

## 汇总：按容量

| hidden | MLP 参数量 | 平均 ΔMSE | 偏好非线性的状态数 | 非线性平均胜率 |
|---|---|---|---|---|
| 23 | 9431 | +0.08422 | 12/108 | 29.2% |
| 64 | 26200 | +0.06194 | 43/108 | 39.7% |
| 128 | 52376 | +0.09128 | 44/108 | 46.5% |

## 检查 (2)：状态间 Range(ΔMSE) 随容量变化

| hidden | 平均状态间极差 | GSPC | BTCUSD | ETHUSD |
|---|---|---|---|---|
| 23 | 0.24487 | 0.10588 | 0.53967 | 0.08907 |
| 64 | 0.26704 | 0.09762 | 0.65525 | 0.04824 |
| 128 | 0.40698 | 0.06016 | 1.12042 | 0.04036 |

## 检查 (1)：符号反转（同一 asset/state 在不同容量下 ΔMSE 变号）

| asset | state | seed | ΔMSE_23 | ΔMSE_64 | ΔMSE_128 |
|---|---|---|---|---|---|
| ETHUSD | acf1_hi | 2021 | +0.01993 | -0.02156 | -0.03421 |
| ETHUSD | acf1_hi | 2022 | +0.00064 | -0.03075 | -0.03862 |
| ETHUSD | acf1_hi | 2023 | +0.01635 | -0.03859 | -0.03153 |
| ETHUSD | acf1_lo | 2021 | +0.01249 | -0.01587 | -0.02452 |
| ETHUSD | acf1_lo | 2023 | +0.00663 | -0.02807 | -0.01438 |
| ETHUSD | jump_ratio_hi | 2021 | +0.00837 | -0.01971 | -0.02927 |
| ETHUSD | jump_ratio_lo | 2021 | +0.02669 | -0.01979 | -0.03298 |
| ETHUSD | jump_ratio_lo | 2022 | +0.00623 | -0.03077 | -0.04040 |
| ETHUSD | jump_ratio_lo | 2023 | +0.03151 | -0.04189 | -0.02982 |
| ETHUSD | kmeans_0 | 2021 | +0.04438 | -0.00957 | -0.02648 |
| ETHUSD | kmeans_0 | 2022 | +0.02470 | -0.01048 | -0.03527 |
| ETHUSD | kmeans_0 | 2023 | +0.04817 | -0.02963 | -0.03358 |
| ETHUSD | kmeans_1 | 2021 | +0.00525 | -0.02443 | -0.03327 |
| ETHUSD | rv_rel_hi | 2021 | +0.00209 | -0.02347 | -0.03040 |
| ETHUSD | rv_rel_lo | 2021 | +0.04426 | -0.01721 | -0.03693 |
| ETHUSD | rv_rel_lo | 2022 | +0.02437 | -0.02452 | -0.05067 |
| ETHUSD | rv_rel_lo | 2023 | +0.05458 | -0.04770 | -0.04872 |
| ETHUSD | sign_persistence_hi | 2021 | +0.05182 | -0.02146 | -0.04197 |
| ETHUSD | sign_persistence_hi | 2022 | +0.03444 | -0.02600 | -0.05474 |
| ETHUSD | sign_persistence_hi | 2023 | +0.06982 | -0.05781 | -0.04762 |
| ETHUSD | sign_persistence_lo | 2021 | +0.00808 | -0.01928 | -0.02813 |
| ETHUSD | structure_strong | 2021 | +0.03859 | -0.01727 | -0.03594 |
| ETHUSD | structure_strong | 2022 | +0.01695 | -0.02651 | -0.04367 |
| ETHUSD | structure_strong | 2023 | +0.04434 | -0.04392 | -0.03656 |
| GSPC | acf1_hi | 2023 | +0.06479 | -0.01050 | -0.01970 |
| GSPC | acf1_lo | 2023 | +0.11703 | +0.00301 | -0.00244 |
| GSPC | jump_ratio_lo | 2023 | +0.08501 | -0.01829 | -0.02890 |
| GSPC | kmeans_2 | 2023 | +0.08244 | -0.01843 | -0.02843 |
| GSPC | rv_rel_hi | 2023 | +0.08124 | -0.00661 | -0.00906 |
| GSPC | rv_rel_lo | 2023 | +0.08813 | -0.00257 | -0.02034 |
| GSPC | sign_persistence_hi | 2023 | +0.09721 | -0.01682 | -0.02660 |
| GSPC | structure_strong | 2023 | +0.09835 | -0.02546 | -0.03549 |

## 检查 (3)：逐 seed 方向一致性

- hidden=23: 31/36 个 asset×state 的 3 个种子方向一致；不一致：ETHUSD/acf1_lo, ETHUSD/jump_ratio_hi, ETHUSD/kmeans_1, ETHUSD/rv_rel_hi, ETHUSD/sign_persistence_lo
- hidden=64: 29/36 个 asset×state 的 3 个种子方向一致；不一致：GSPC/acf1_hi, GSPC/jump_ratio_lo, GSPC/kmeans_2, GSPC/rv_rel_hi, GSPC/rv_rel_lo, GSPC/sign_persistence_hi, GSPC/structure_strong
- hidden=128: 28/36 个 asset×state 的 3 个种子方向一致；不一致：GSPC/acf1_hi, GSPC/acf1_lo, GSPC/jump_ratio_lo, GSPC/kmeans_2, GSPC/rv_rel_hi, GSPC/rv_rel_lo, GSPC/sign_persistence_hi, GSPC/structure_strong

## 结论（按协议 §4 三路诊断）

- 128 维下有 40.7% 的状态偏好非线性。是否构成 Case 2（跨 seed、CI 支持、非单资产）需对照上表逐项核对；仅 ETH 成立则为 Case 3（asset-specific conditional specialization）。
