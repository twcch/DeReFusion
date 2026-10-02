# 容量敏感性检查（Capacity Sensitivity）

隐层宽度 ∈ [23, 64, 128]；其余协议与主实验完全一致；linear 算子与宽度无关（同种子复用）。

## 汇总：按容量

| hidden | MLP 参数量 | 平均 ΔMSE | 偏好非线性的状态数 | 非线性平均胜率 |
|---|---|---|---|---|
| 23 | 9431 | +0.20327 | 21/264 | 27.4% |
| 64 | 26200 | +0.12768 | 63/264 | 39.7% |
| 128 | 52376 | +0.06415 | 105/264 | 49.8% |

## 检查 (2)：状态间 Range(ΔMSE) 随容量变化

| hidden | 平均状态间极差 | USDJPY | EURUSD | SOX | DJI | BABA | NVO | TM |
|---|---|---|---|---|---|---|---|---|
| 23 | 0.46231 | 0.32532 | 0.20065 | 0.19969 | 0.10202 | 0.06054 | 1.98382 | 0.36411 |
| 64 | 0.41260 | 0.13139 | 0.23185 | 0.23328 | 0.07096 | 0.04743 | 2.10125 | 0.07207 |
| 128 | 0.43524 | 0.16741 | 0.30190 | 0.16632 | 0.09254 | 0.08546 | 2.05402 | 0.17902 |

## 检查 (1)：符号反转（同一 asset/state 在不同容量下 ΔMSE 变号）

| asset | state | seed | ΔMSE_23 | ΔMSE_64 | ΔMSE_128 |
|---|---|---|---|---|---|
| BABA | acf1_hi | 2022 | +0.00343 | -0.00641 | -0.00046 |
| BABA | acf1_hi | 2023 | +0.00169 | +0.00064 | -0.00299 |
| BABA | acf1_lo | 2021 | -0.00544 | -0.00139 | +0.00009 |
| BABA | acf1_lo | 2022 | -0.00277 | -0.00350 | +0.00595 |
| BABA | jump_ratio_hi | 2021 | +0.00584 | -0.00325 | -0.00420 |
| BABA | jump_ratio_hi | 2022 | -0.00134 | -0.00820 | +0.00115 |
| BABA | jump_ratio_hi | 2023 | -0.00367 | +0.00213 | -0.00344 |
| BABA | jump_ratio_lo | 2022 | -0.00016 | -0.00271 | +0.00492 |
| BABA | kmeans_1 | 2022 | -0.00229 | -0.00545 | +0.00314 |
| BABA | kmeans_2 | 2022 | +0.00737 | -0.00038 | +0.00604 |
| BABA | rv_rel_hi | 2022 | +0.00100 | -0.01301 | +0.00440 |
| BABA | rv_rel_hi | 2023 | +0.00046 | +0.00709 | -0.00702 |
| BABA | rv_rel_lo | 2021 | -0.01346 | +0.00265 | +0.01843 |
| BABA | rv_rel_lo | 2022 | -0.00479 | +0.01323 | +0.00313 |
| BABA | sign_persistence_hi | 2021 | +0.00299 | -0.01057 | -0.01274 |
| BABA | sign_persistence_hi | 2022 | +0.00606 | -0.00460 | -0.00276 |
| BABA | sign_persistence_hi | 2023 | +0.00125 | -0.00220 | -0.00521 |
| BABA | sign_persistence_lo | 2022 | -0.00228 | -0.00453 | +0.00533 |
| BABA | structure_strong | 2022 | +0.00250 | -0.01324 | -0.00240 |
| BABA | structure_weak | 2021 | -0.00705 | +0.00221 | +0.00551 |
| BABA | structure_weak | 2022 | -0.00274 | +0.00169 | +0.00801 |
| DJI | acf1_lo | 2021 | -0.01871 | +0.01501 | +0.04115 |
| DJI | acf1_lo | 2022 | +0.08331 | +0.07165 | -0.00751 |
| DJI | rv_rel_lo | 2021 | -0.01000 | +0.01639 | +0.04328 |
| DJI | rv_rel_lo | 2022 | +0.06470 | +0.05610 | -0.00254 |
| NVO | acf1_hi | 2022 | +0.36773 | +0.25006 | -0.07038 |
| NVO | acf1_hi | 2023 | +0.30641 | +0.14483 | -0.09209 |
| NVO | acf1_lo | 2022 | +0.78080 | +0.46896 | -0.03294 |
| NVO | acf1_lo | 2023 | +0.59881 | +0.44918 | -0.11537 |
| NVO | jump_ratio_hi | 2023 | +0.97502 | +1.08989 | -0.13893 |
| NVO | jump_ratio_lo | 2022 | +0.50590 | +0.29744 | -0.06481 |
| NVO | jump_ratio_lo | 2023 | +0.39964 | +0.16227 | -0.10156 |
| NVO | kmeans_0 | 2022 | +0.42750 | +0.20851 | -0.09687 |
| NVO | kmeans_0 | 2023 | +0.32114 | +0.21990 | -0.14814 |
| NVO | kmeans_1 | 2022 | +0.92715 | +0.63149 | -0.04774 |
| NVO | kmeans_1 | 2023 | +0.74035 | +0.41476 | -0.13720 |
| NVO | rv_rel_hi | 2022 | +0.90302 | +0.56925 | -0.02814 |
| NVO | rv_rel_hi | 2023 | +0.69394 | +0.54537 | -0.12509 |
| NVO | rv_rel_lo | 2022 | +0.29827 | +0.14155 | -0.09953 |
| NVO | rv_rel_lo | 2023 | +0.24892 | +0.11542 | -0.11881 |
| NVO | sign_persistence_hi | 2022 | +0.87111 | +0.54401 | -0.04646 |
| NVO | sign_persistence_hi | 2023 | +0.65337 | +0.58216 | -0.16006 |
| NVO | sign_persistence_lo | 2022 | +0.46462 | +0.26537 | -0.03282 |
| NVO | sign_persistence_lo | 2023 | +0.38739 | +0.12815 | -0.04508 |
| NVO | structure_strong | 2022 | +0.55043 | +0.31143 | -0.09710 |
| NVO | structure_strong | 2023 | +0.41658 | +0.33954 | -0.15538 |
| NVO | structure_weak | 2023 | +0.67373 | +0.43906 | -0.06132 |
| SOX | acf1_hi | 2021 | +0.04462 | -0.00768 | -0.01169 |
| SOX | acf1_hi | 2022 | +0.10806 | -0.00196 | -0.03850 |
| SOX | acf1_hi | 2023 | +0.13371 | -0.03935 | -0.06007 |
| SOX | acf1_lo | 2023 | +0.13732 | +0.00105 | -0.02138 |
| SOX | jump_ratio_hi | 2023 | +0.17118 | +0.00589 | -0.01596 |
| SOX | jump_ratio_lo | 2021 | +0.03785 | +0.00152 | -0.00986 |
| SOX | jump_ratio_lo | 2022 | +0.10356 | +0.01152 | -0.02830 |
| SOX | jump_ratio_lo | 2023 | +0.12411 | -0.02721 | -0.04869 |
| SOX | kmeans_1 | 2021 | +0.06484 | -0.00405 | -0.00543 |
| SOX | kmeans_1 | 2022 | +0.19073 | +0.02808 | -0.02381 |
| SOX | kmeans_1 | 2023 | +0.21846 | -0.02294 | -0.05767 |
| SOX | kmeans_2 | 2021 | +0.03136 | +0.00378 | -0.01250 |
| SOX | kmeans_2 | 2022 | +0.09245 | +0.01429 | -0.02531 |
| SOX | kmeans_2 | 2023 | +0.10834 | -0.03046 | -0.04835 |
| SOX | rv_rel_hi | 2023 | +0.08395 | +0.00578 | -0.01435 |
| SOX | rv_rel_lo | 2021 | +0.05839 | +0.00448 | -0.01917 |
| SOX | rv_rel_lo | 2022 | +0.20448 | +0.03832 | -0.03845 |
| SOX | rv_rel_lo | 2023 | +0.23105 | -0.06111 | -0.08637 |
| SOX | sign_persistence_hi | 2021 | +0.05101 | +0.01266 | -0.02282 |
| SOX | sign_persistence_hi | 2022 | +0.20355 | +0.05034 | -0.03198 |
| SOX | sign_persistence_hi | 2023 | +0.22512 | -0.06327 | -0.08697 |
| SOX | sign_persistence_lo | 2022 | +0.05931 | +0.01725 | -0.00007 |
| SOX | sign_persistence_lo | 2023 | +0.07608 | +0.01004 | -0.01013 |
| SOX | structure_strong | 2021 | +0.04562 | +0.01107 | -0.02593 |
| SOX | structure_strong | 2022 | +0.17464 | +0.04325 | -0.03095 |
| SOX | structure_strong | 2023 | +0.19115 | -0.06313 | -0.08442 |
| SOX | structure_weak | 2023 | +0.09207 | +0.01511 | -0.00668 |
| TM | acf1_hi | 2021 | +0.12727 | -0.00624 | -0.03927 |
| TM | acf1_hi | 2022 | +0.06385 | -0.01201 | -0.07835 |
| TM | acf1_hi | 2023 | +0.13898 | -0.00073 | -0.01904 |
| TM | acf1_lo | 2021 | +0.01725 | -0.00471 | -0.01697 |
| TM | acf1_lo | 2022 | +0.00365 | -0.02778 | -0.02795 |
| TM | acf1_lo | 2023 | +0.01825 | -0.02653 | -0.04020 |
| TM | jump_ratio_hi | 2021 | +0.11925 | +0.00058 | -0.06292 |
| TM | jump_ratio_hi | 2022 | +0.09859 | -0.05657 | -0.11352 |
| TM | jump_ratio_hi | 2023 | +0.19839 | -0.06204 | -0.13101 |
| TM | jump_ratio_lo | 2021 | +0.05896 | -0.00761 | -0.01669 |
| TM | jump_ratio_lo | 2022 | +0.01295 | -0.00681 | -0.03364 |
| TM | kmeans_0 | 2021 | +0.04997 | -0.00640 | -0.01753 |
| TM | kmeans_0 | 2022 | +0.01059 | -0.00934 | -0.03212 |
| TM | kmeans_0 | 2023 | +0.03048 | -0.00710 | -0.00906 |
| TM | kmeans_1 | 2021 | +0.25493 | +0.00912 | -0.07092 |
| TM | kmeans_1 | 2022 | +0.17349 | -0.02988 | -0.16839 |
| TM | kmeans_1 | 2023 | +0.36710 | +0.00562 | -0.07340 |
| TM | kmeans_2 | 2021 | +0.03393 | -0.01292 | -0.03900 |
| TM | kmeans_2 | 2022 | +0.02483 | -0.05065 | -0.05317 |
| TM | kmeans_2 | 2023 | +0.06106 | -0.04961 | -0.07260 |
| TM | rv_rel_hi | 2021 | +0.09004 | +0.00083 | -0.02084 |
| TM | rv_rel_hi | 2022 | +0.03643 | -0.00830 | -0.05371 |
| TM | rv_rel_hi | 2023 | +0.08854 | +0.00126 | -0.02138 |
| TM | rv_rel_lo | 2021 | +0.01198 | -0.02990 | -0.04300 |
| TM | rv_rel_lo | 2022 | +0.02646 | -0.02895 | -0.03416 |
| TM | rv_rel_lo | 2023 | +0.04768 | -0.03359 | -0.01458 |
| TM | sign_persistence_hi | 2021 | +0.20348 | +0.00166 | -0.05828 |
| TM | sign_persistence_hi | 2022 | +0.10571 | -0.03372 | -0.13466 |
| TM | sign_persistence_hi | 2023 | +0.24181 | -0.01115 | -0.06116 |
| TM | sign_persistence_lo | 2021 | +0.01618 | -0.00874 | -0.01515 |
| TM | sign_persistence_lo | 2022 | +0.00299 | -0.01320 | -0.01780 |
| TM | sign_persistence_lo | 2023 | +0.00841 | -0.01401 | -0.01477 |
| TM | structure_strong | 2021 | +0.10120 | -0.00757 | -0.02770 |
| TM | structure_strong | 2022 | +0.03157 | -0.00543 | -0.05368 |
| TM | structure_weak | 2021 | +0.05034 | -0.00365 | -0.02934 |
| TM | structure_weak | 2022 | +0.03798 | -0.03234 | -0.05457 |
| TM | structure_weak | 2023 | +0.08063 | -0.03398 | -0.06507 |

## 检查 (3)：逐 seed 方向一致性

- hidden=23: 75/88 个 asset×state 的 3 个种子方向一致；不一致：BABA/acf1_hi, BABA/acf1_lo, BABA/jump_ratio_hi, BABA/jump_ratio_lo, BABA/kmeans_1, BABA/kmeans_2, BABA/rv_rel_hi, BABA/rv_rel_lo, BABA/sign_persistence_lo, BABA/structure_strong, BABA/structure_weak, DJI/acf1_lo, DJI/rv_rel_lo
- hidden=64: 68/88 个 asset×state 的 3 个种子方向一致；不一致：BABA/acf1_hi, BABA/acf1_lo, BABA/jump_ratio_hi, BABA/jump_ratio_lo, BABA/kmeans_1, BABA/kmeans_2, BABA/rv_rel_hi, BABA/sign_persistence_lo, SOX/jump_ratio_lo, SOX/kmeans_1, SOX/kmeans_2, SOX/rv_rel_lo, SOX/sign_persistence_hi, SOX/structure_strong, TM/jump_ratio_hi, TM/jump_ratio_lo, TM/kmeans_1, TM/rv_rel_hi, TM/sign_persistence_hi, TM/structure_strong
- hidden=128: 61/88 个 asset×state 的 3 个种子方向一致；不一致：BABA/jump_ratio_hi, BABA/jump_ratio_lo, BABA/kmeans_1, BABA/kmeans_2, BABA/rv_rel_hi, BABA/sign_persistence_lo, DJI/acf1_lo, DJI/rv_rel_lo, NVO/acf1_hi, NVO/acf1_lo, NVO/jump_ratio_hi, NVO/jump_ratio_lo, NVO/kmeans_0, NVO/kmeans_1, NVO/rv_rel_hi, NVO/rv_rel_lo, NVO/sign_persistence_hi, NVO/sign_persistence_lo, NVO/structure_strong, NVO/structure_weak, SOX/acf1_lo, SOX/jump_ratio_hi, SOX/rv_rel_hi, SOX/sign_persistence_lo, SOX/structure_weak, TM/jump_ratio_lo, TM/structure_strong

## 结论（按协议 §4 三路诊断）

- 128 维下有 39.8% 的状态偏好非线性。是否构成 Case 2（跨 seed、CI 支持、非单资产）需对照上表逐项核对；仅 ETH 成立则为 Case 3（asset-specific conditional specialization）。
