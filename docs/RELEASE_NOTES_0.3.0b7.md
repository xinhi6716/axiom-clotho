# Axiom Clotho 0.3.0b7 — Celestial visual maintenance

This candidate packages the Clotho visual and documentation refresh on top of
source commit `8f9a7e1a91d30bca2cd566ab73d48d69a8d77faa`. It is **cosmetic and
documentation maintenance**, not a new ML capability, accuracy improvement or
scientific-validation release. A local directory, version label or this guide
does not establish publication.

## What changes from b6

- A recognizable spindle emblem, open celestial framing and coordinated ivory,
  midnight and gold artwork across light, dark and compact README heroes.
- The same frameless spindle in regression and binary HTML report headers,
  with coordinated brand colors and accessible small-text gold. Report data,
  calculations, chart meaning, language controls and security policy are unchanged.
- README process diagrams recolored without changing any words, geometry or
  data semantics. English-first and Traditional Chinese landing pages keep the
  practical installation, input, output and limitation details.
- Three original upstream tool guides restored by the source commit above:
  I–V read points, RRAM read windows and device-condition fields. They retain
  their upstream text, pinned provenance and MIT license; upstream-root/Rust
  commands inside those guides are not new Clotho CLI routes.

The [b6](RELEASE_NOTES_0.3.0b6.md) regression finite-tree-state guard is retained
unchanged. Model defaults, candidate grids, preprocessing, group splits, metric
formulas, result schemas, dependency bounds and workflow contracts are unchanged.
No real dataset, model weights, new estimator family or scientific result is
added. Blank-input `init` from [b4](RELEASE_NOTES_0.3.0b4.md) and FET from
[b3](RELEASE_NOTES_0.3.0b3.md) remain available. All historical release guides,
tags and assets, including b6, remain unchanged; b6 downloads do not contain
this newer visual refresh.

## Installation and reproducible example

Before following a published-release route, confirm that the matching wheel,
source archive and `SHA256SUMS.txt` exist at the
[b7 release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b7).
For a privately supplied candidate, verify complete local-file SHA-256 values
against its accompanying checksum file. Use a separate environment with
compatible dependencies already available:

```sh
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b7-py3-none-any.whl
clotho --version
mlcalc-workflow --version
mlcalc --version
groupbench --version
clotho components
clotho init --workflow rram --out my-measurements
# OR, using another fresh directory:
clotho init --workflow fet --out my-transfer-data
```

All four version commands should report `0.3.0b7`. Init help/checklists still
identify b4 as the introduction version, not the installed package version.
The offline install does not fetch missing dependencies. Requirements remain
Python >=3.10, NumPy >=1.24,<3, pandas >=2,<3, scikit-learn >=1.4,<2 and
threadpoolctl >=3,<4; these bounds are not a tested version matrix. The
`ml-super-calculator` distribution and four CLI names retain one owner. Do not
co-install an overlapping distribution. No PyPI/GitHub Packages publication is
implied.

Use the matching source archive for the unchanged invented
[RRAM](../examples/measurement-regression/README.md) and
[FET](../examples/fet-device-regression/README.md) fixtures. Runtime code,
pinned components and licenses are in the wheel; tests, example fixtures and
documentation are additionally in the source archive. The CLI downloads none.
These are software fixtures, not real-device performance results or recommended
scientific-study settings.

## Verification and limits

The exact integrated visual/documentation source, still labeled b6 before this
packaging step, completed **615 tests: 609 passed, 6 skipped**, with zero
failures/errors. The six unchanged cross-repository semiconductor-toolbox tests
were skipped because a separate trusted matching checkout was not supplied;
they were not passed. This is inherited source evidence, not a fresh b7 full
suite or a test of the older published b6 wheel.

The additional b7 delta changes version constants, the matching version test
and current release guidance only. The full 615-test suite is **not rerun for
b7 packaging**. Separate focused checks cover offline final-wheel installation,
CLI versions, installed report-emblem/reading-path contracts, archive payloads,
licenses, wheel RECORD integrity, two clean local reproducible builds and a
wheel rebuilt from the source archive. Exact counts, artifact identities and
measured costs belong to the accompanying verification receipt. Overlapping
source and package checks must not be added together.

Checks use Linux/Python 3.12.14 with existing NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0 and threadpoolctl 3.6.0. A disposable offline environment
reuses provisioned dependencies, not freshly resolved dependencies. Native
Windows/macOS, Python 3.10, minimum-dependency combinations and hosted CI remain
unverified; a `py3-none-any` tag is not native-platform validation. Complete
RRAM/FET measurement routes are not rerun for this packaging step. Static SVG
context and source-contract checks are not full browser, assistive-technology
or printer validation. Local checks do not establish remote downloaded-binary
verification or publication.

## Filesystem safety, licensing and privacy

Use init only in **trusted, idle local directories**. Failed creation can leave
partial output; retry a fresh name. A status file is not proof of success.
This is not a filesystem transaction or comprehensive TOCTOU defense: hostile
concurrent ancestor replacement can create outside files before detection.
See the [setup guide](WORKFLOW_INIT.md). This release does not alter that limit.

Original code and invented fixtures remain MIT-licensed; bundled code, artwork
and outlined typography retain their notices. No user measurements, telemetry,
automatic upload or hosted ML service is added. Review filled inputs and reports
before sharing. Passing software checks does not establish real-device quality,
physical validity, causal identification or calibrated uncertainty.

## 繁體中文：版本與驗證範圍

b7 更新紡錘圖像、明暗與行動版 README 視覺、報告徽記及配色，並整理英文優先
與繁體中文文件。流程圖只換色，不改文字、幾何或資料意義；三份工具原文保留
固定來源與授權。這不是新的 ML 能力、準確度提升或科學驗證。

b6 的迴歸有限樹狀態防護原樣保留；模型、候選格點、前處理、切分、指標、綱要
與相依套件不變。所有歷史指南、tag 與下載檔維持原樣，包括不含本次新視覺的 b6。

整合後但尚標示 b6 的精確原始碼已完成 615 項測試：609 通過、6 項外部整合略過。
這是繼承的原始碼證據，不是 b7 重跑全套，也不是舊 b6 wheel 的測試。b7 另做
離線安裝、版本與報告、封包內容、校驗碼及可重現建置檢查；精確結果見隨附收據，
重疊項目不可相加。原生平台、最低相依版本矩陣、完整瀏覽器與列印及託管 CI
仍未驗證。候選檔不等於發布；安裝前請核對完整 SHA-256 與四個版本指令。
