# Axiom Clotho 0.3.0b6 — Regression finite-tree-state hardening

This candidate packages the reviewed regression finite-tree-state guard from
source commit `0186e1299621233c58884aedbb636b9e0c9be6c2`. It is
**numerical-integrity hardening**, not a new ML capability, accuracy improvement
or scientific-validation release. A candidate directory, version number or this
guide does not establish publication.

## What changes from b5

Immediately after each inner fit and family refit, supported regression
`ExtraTreesRegressor` models are checked for non-finite stored `threshold`,
`value`, `impurity` and `weighted_n_node_samples` arrays in every tree. Invalid
state raises `ValueError` before prediction; it is not clipped, silently
rescaled, retried or skipped. Failed refits retain prior completed estimator
state. The regression `run` and `predict` CLIs report failure without creating
or replacing output in the covered synthetic case, including with `--overwrite`.

The motivating fixture is deliberately extreme and invented: 16 rows with the
constant target `float(2**600)`. Its predictions and error metrics were finite
and exact despite NaN tree impurity. This is **not evidence of ordinary-data
prediction failure**. The ordinary constant-target control remains unchanged.

This is a post-fit boundary for the supported regression ExtraTrees family,
not a recursive audit of arbitrary estimators, a binary/classification guard or
protection against later direct caller mutation of fitted state. Model defaults,
candidate grids, preprocessing, group splits, metric formulas, result schemas,
dependency bounds and existing workflow contracts are unchanged. It adds no
real dataset, model weights, estimator family or scientific result.

The b5 artwork and report presentation are retained byte-for-byte; no new Moirai
artwork is included. Blank-input `init` from [b4](RELEASE_NOTES_0.3.0b4.md) and FET
from [b3](RELEASE_NOTES_0.3.0b3.md) remain. Frozen [b5](RELEASE_NOTES_0.3.0b5.md)
source `dbf5d952307d84461f0303abe00e8c442efe38f5` and its wheel/source assets
**do not contain this guard** and are not replaced. All earlier release guides,
tags and assets remain unchanged.

## Installation and reproducible example

Before using a published-release route, confirm the matching wheel, source
archive and `SHA256SUMS.txt` exist at the
[b6 release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b6).
For a privately supplied candidate, verify complete local-file SHA-256 values
against its accompanying checksum file. Use a separate environment with
compatible dependencies already available:

```sh
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b6-py3-none-any.whl
clotho --version
mlcalc-workflow --version
mlcalc --version
groupbench --version
clotho components
clotho init --workflow rram --out my-measurements
# OR, using another fresh directory:
clotho init --workflow fet --out my-transfer-data
```

All four version commands should report `0.3.0b6`. Init help/checklists still
identify b4 as its introduction version, not the installed package version.
The offline install does not fetch missing dependencies. Requirements remain
Python >=3.10, NumPy >=1.24,<3, pandas >=2,<3, scikit-learn >=1.4,<2 and
threadpoolctl >=3,<4; these bounds are not a tested version matrix. The
`ml-super-calculator` distribution and four CLI names retain one owner. Do not
co-install an overlapping distribution. No PyPI/GitHub Packages publication is
implied.

Use the matching source archive for the unchanged invented
[RRAM](../examples/measurement-regression/README.md) and
[FET](../examples/fet-device-regression/README.md) examples. Runtime code, pinned
components and licenses are in the wheel; tests, example fixtures and
documentation are additionally in the source archive. The CLI downloads none.
These are software fixtures, not real-device performance results or recommended
scientific-study settings.

## Verification and limits

The accepted guard source at the exact commit above completed **609 tests:
603 passed, 6 skipped**, with zero failures/errors, before this b6 packaging
change. The six unchanged external semiconductor-toolbox walkthrough tests were
skipped because a separate trusted matching checkout was not supplied. They
were not passed. Independent review accepted that exact two-file source patch.

This b6 package changes version constants and one matching version test, plus
release documentation; the guard implementation and its nine tests are unchanged
from that accepted source. The full suite was **not rerun for b6**. Packaging
verification is a separate focused check, not a new complete-suite result:
installed-wheel guard and version checks, offline documentation-contract tests,
archive payload/license/RECORD checks and two clean local reproducible builds.
Exact counts, hashes and measured costs belong to the candidate's accompanying
verification receipt. Earlier source-suite and focused checks overlap and must
not be added together.

Checks use Linux/Python 3.12.14 with existing NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0 and threadpoolctl 3.6.0. An offline disposable environment
reuses provisioned dependencies; this is not fresh dependency resolution.
Native Windows/macOS, Python 3.10, minimum-dependency combinations and hosted
CI are unverified. A `py3-none-any` tag is not native-platform validation.
The complete RRAM/FET measurement routes and browser/assistive/print rendering
are not rerun for this packaging change. Local artifact checks do not establish
remote downloaded-binary verification or publication.

## Filesystem safety, licensing and privacy

Use init only in **trusted, idle local directories**. Failed creation can leave
partial output; retry a fresh name. A status file is not proof of success. This
is not a filesystem transaction or comprehensive TOCTOU defense: hostile
concurrent ancestor replacement can create outside files before detection. See
the [setup guide](WORKFLOW_INIT.md). This guard does not alter that limitation.

Original code and invented fixtures remain MIT-licensed; existing bundled code,
artwork and typography retain their notices. No user measurements, telemetry,
automatic upload or hosted ML service is added. Review filled inputs and reports
before sharing. Passing software checks does not establish real-device quality,
physical validity, causal identification or calibrated uncertainty.

## 繁體中文：版本與驗證範圍

b6 候選版本封裝迴歸 ExtraTrees 的有限樹狀態防護：每次內層擬合及模型家族
重擬合後，逐棵檢查 threshold、value、impurity 與 weighted_n_node_samples。
遇到非有限值便在預測前失敗，不截斷、不縮放、不重試、不跳過候選模型；
失敗的重擬合保留先前完整狀態。這不是新的 ML 能力或科學驗證。

動機案例是刻意極端、完全虛構的 16 列常數目標 `float(2**600)`；其預測及
誤差原本仍有限且精確，問題是樹內 NaN impurity，不應描述成一般資料預測錯誤。
防護不涵蓋任意模型狀態、二元分類或使用者之後直接修改已擬合模型。
模型預設、候選格點、前處理、群組切分、指標公式、綱要與相依套件不變。

既有已接受防護原始碼的完整測試是 603 通過、6 略過（共 609），不是 b6
重新執行的完整結果。b6 只另外驗證離線安裝、相關防護與版本、文件及封裝；
精確項數、雜湊與成本見隨附收據。不同驗證範圍重疊，不可相加；六項外部
整合略過不算通過。原生平台、最低相依版本矩陣及託管 CI 均未驗證。

b5 視覺、b4 引入的 init、b3 引入的 FET 均保留；沒有加入新 Moirai 素材。
b5 的固定原始碼與下載資產不含本次防護，也不會被替換。安裝 b6 前確認
相符檔案與完整 SHA-256；四個版本指令應回報 `0.3.0b6`，候選檔不等於發布。
只在可信、閒置本機目錄操作；既有 TOCTOU、隱私與科學限制不變。
