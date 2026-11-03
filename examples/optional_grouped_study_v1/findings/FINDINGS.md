# Optional grouped models: synthetic findings / 可選分組模型：合成結果

**Conclusion / 結論：** Keep the nonlinear kernel optional, not the default. It helped on the registered smooth IID mechanism but failed badly on both extrapolation folds of the linear domain-holdout mechanism. The median baseline showed no consistent advantage over dummy. / 保留核模型為可選項，不改預設；它改善了指定平滑 IID 機制，但在線性跨域的左右外插折均明顯失利。分組中位數未展現對 dummy 的一致優勢。

## Frozen experiment / 固定實驗

Five roots × three scenarios; 18 groups and 54 rows per dataset; outer 3 / inner 2 group-disjoint splits; five fixed families plus inner-selected output. All 15 datasets and 1,125 fits completed with zero fit failures. All results retained; no refits, retuning, resampling, or excluded seeds for this report. Primary metric: equally weighted group MAE; secondary: row MAE (CSV column `mae`). Lower is better. / 五個種子 × 三個情境，每組資料 18 群、54 列，外層 3 折、內層 2 折；五模型及內層選模輸出全部保留。此報告沒有重訓、調參、重抽樣或剔除種子。

Roots: 731903, 731927, 731939, 731951, 731967. Protocol SHA-256: `708227ab3f634ffaed0ab5179e9189e4ea4f6019ea7e8c8e8eb159adcf41556a`.

## All models, scenario summaries / 全模型情境摘要

Means over five roots; brackets show seed minimum–maximum, **not confidence intervals**. Domain-holdout pooled values mix two extrapolation folds and one interpolation fold. / 五種子平均與範圍，並非信賴區間；跨域合併值混合兩個外插折及一個內插折。

| Scenario | Model | Group MAE mean [min–max] | Row MAE mean [min–max] |
|---|---|---:|---:|
| linear_iid | dummy | 1.07071 [0.96111–1.19201] | 1.08465 [0.98976–1.20796] |
| linear_iid | ridge | 0.20780 [0.18047–0.23592] | 0.21013 [0.18638–0.24193] |
| linear_iid | extra_trees | 0.26863 [0.23786–0.28924] | 0.27340 [0.24973–0.29560] |
| linear_iid | nystroem_ridge | 0.27007 [0.24589–0.29525] | 0.27191 [0.25466–0.29854] |
| linear_iid | group_median | 1.09870 [0.96369–1.27945] | 1.10967 [0.99027–1.28781] |
| linear_iid | selected | 0.20780 [0.18047–0.23592] | 0.21013 [0.18638–0.24193] |
| smooth_iid | dummy | 1.26038 [1.13062–1.46043] | 1.27438 [1.13028–1.50309] |
| smooth_iid | ridge | 0.82588 [0.75541–0.88937] | 0.81107 [0.71898–0.88163] |
| smooth_iid | extra_trees | 0.36641 [0.29818–0.41885] | 0.35998 [0.28328–0.40844] |
| smooth_iid | nystroem_ridge | 0.30265 [0.28499–0.32918] | 0.29724 [0.28644–0.30721] |
| smooth_iid | group_median | 1.28091 [1.12619–1.55097] | 1.29293 [1.09745–1.59471] |
| smooth_iid | selected | 0.29953 [0.27631–0.32412] | 0.29475 [0.26428–0.30909] |
| linear_domain_holdout | dummy | 3.16316 [2.93836–3.25728] | 3.17441 [2.93257–3.25992] |
| linear_domain_holdout | ridge | 0.21759 [0.12914–0.27815] | 0.21542 [0.13291–0.28472] |
| linear_domain_holdout | extra_trees | 1.61598 [1.48078–1.84884] | 1.62516 [1.47987–1.84525] |
| linear_domain_holdout | nystroem_ridge | 2.39674 [2.24783–2.57990] | 2.40908 [2.26921–2.59266] |
| linear_domain_holdout | group_median | 3.45647 [3.31292–3.64113] | 3.45888 [3.31065–3.63834] |
| linear_domain_holdout | selected | 0.41599 [0.12914–1.19899] | 0.42388 [0.13291–1.24355] |

## Interpretation / 解讀

- **Linear IID:** ridge mean group MAE 0.20780 versus kernel 0.27007. Kernel lost to ridge in all five seeds; selected matched ridge in all five. / 線性 IID 下 ridge 較佳，kernel 五種子全敗，選模輸出與 ridge 相同。
- **Smooth IID:** kernel 0.30265 versus extra trees 0.36641 and ridge 0.82588. Kernel improved over extra trees in 4/5 seeds and ridge in 5/5; this is a useful, mechanism-specific nonlinear gain, not universal superiority. / 平滑 IID 下核模型有實際改善，但僅限此機制，並非普遍勝出。
- **Domain holdout:** pooled ridge 0.21759, kernel 2.39674, selected 0.41599. Kernel was worse than ridge **and extra trees in 5/5 seeds on each extrapolation side**. Kernel minus ridge mean group-MAE differences: left +3.00002, right +3.49821, interpolation +0.03924 (worse in 4/5 there). Selected matched ridge in four seeds but lost in one due to the left extrapolation fold: selection did not reliably protect against extrapolation failure. / 跨域結果必須分開看：左右外插各五種子均失利；內插差異較小。選模也不能保證避開外插失敗。
- **Median versus dummy:** linear IID mean difference +0.02798 (worse 4/5); smooth IID +0.02053 (improves 3/5 yet worse on average); mixed-domain +0.29331 (worse 5/5). Domain roles differ: left −0.40987 (improves 5/5), center +1.10201 (worse 5/5), right +0.18779 (worse 4/5). This contrast changes both the location rule and weighting; it does not isolate a benefit of group weighting. / 中位數結果負面或混合，不應隱藏；與 dummy 比較同時改變位置估計法及權重，不能歸因於群權重本身。

## Complete results and limits / 完整結果與限制

`ALL_RESULTS.csv` contains all 360 records: 15 datasets × 6 outputs × (pooled + 3 folds), including group MAE, row MAE, RMSE, worst-group MAE, all seed IDs and fold roles. The pooled records repeat the same held-out observations summarized in folds, not additional independent samples. / CSV 保留所有模型、種子、折角色及指標；合併列不是額外獨立樣本。

Five seeds are descriptive repeats; rows and overlapping-training folds are not independent replicates. There is no cross-scenario universal ranking, significance claim, generic OOD claim, or evidence of real semiconductor/material/device efficacy. Results suggest keeping defaults unchanged and documenting the optional kernel's interpolation-versus-extrapolation limitations, not changing settings after seeing outcomes. / 不宣稱統計顯著、一般化 OOD 能力或真實半導體效益；建議維持預設並清楚標註適用限制。

Terminal supervisor receipt reports 15.623958 child CPU seconds, 15.683630 wall seconds, and child terminal maximum RSS 136640 KiB. However sampled peak RSS is recorded as zero: this is an observation gap caused by unavailable process-topology telemetry bypassing the live checks, **not zero memory use or fully verified real-time memory-budget enforcement**. Managed artifact/source I/O excludes dependency-loader traffic. / 終端資源數據可引用，但即時 RSS 取樣為零是觀測缺口，不能宣稱零記憶體或即時預算已完整驗證。

Evidence: frozen `../independent_qa.json` (independent saved-output arithmetic, no refit), `../artifacts/ALL_CASES_SEALED.json`, and `../artifacts/SUPERVISOR.json`. This rendering only uses saved QA results; source/model code is not imported.
