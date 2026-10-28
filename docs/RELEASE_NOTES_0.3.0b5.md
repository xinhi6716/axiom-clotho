# Axiom Clotho 0.3.0b5 — Report and documentation visual maintenance

Clotho b5 packages the celestial-thread identity already present in current
source: matching coil emblems in regression and binary HTML reports, coordinated
workflow diagrams, English/Traditional Chinese dark and mobile variants, and
guide navigation. This is **visual and documentation maintenance**, not a new ML
capability or a scientific-validation release.

## What changes from b4

The downloadable package brings report presentation into line with the current
documentation. The source archive includes the updated documentation artwork
and navigation. Models, schemas, splits, metrics, dependency requirements,
measurement behavior and invented fixtures are unchanged. The distribution
remains `ml-super-calculator`; all four existing CLI names remain available.

Blank-input `clotho init` was introduced in [b4](RELEASE_NOTES_0.3.0b4.md) and is
retained unchanged. [b3](RELEASE_NOTES_0.3.0b3.md) includes FET but not init; b2
includes neither. Old guides, tags and release assets stay frozen. This package
does not add real data, model weights, estimator families or scientific results.

## Installation and reproducible example

Before using this release route, confirm that the matching wheel, source archive
and `SHA256SUMS.txt` are published at the
[b5 release](https://github.com/xinhi6716/axiom-clotho/releases/tag/v0.3.0b5).
A candidate directory or these instructions alone do not establish publication.
Compare complete local-file SHA-256 values against the matching checksum file.
Use a separate environment with compatible dependencies already available:

```sh
python -m pip install --no-index --no-deps ./ml_super_calculator-0.3.0b5-py3-none-any.whl
clotho --version
mlcalc-workflow --version
mlcalc --version
groupbench --version
clotho components
clotho init --workflow rram --out my-measurements
# OR, using another fresh directory:
clotho init --workflow fet --out my-transfer-data
```

All four version commands report `0.3.0b5`. Init help/checklists still identify
b4 as its introduction version; this is historical capability information, not
the installed package version. The offline command does not fetch missing
dependencies. Requirements remain Python >=3.10, NumPy >=1.24,<3, pandas >=2,<3,
scikit-learn >=1.4,<2 and threadpoolctl >=3,<4. These declared bounds are not a
tested version matrix. No PyPI or GitHub Packages publication is implied.
Do not co-install an overlapping distribution.

Extract the matching b5 source archive for the existing invented
[RRAM](../examples/measurement-regression/README.md) and
[FET](../examples/fet-device-regression/README.md) examples. Runtime code, pinned
components and licenses are in the wheel; tests, example fixtures, generators
and documentation are additionally in the source archive. The CLI downloads
none. These examples are software fixtures, not real-device performance results
or recommended settings for a scientific study.

## Verification and limits

Author verification used Linux/Python 3.12.14 with NumPy 2.3.5, pandas 2.2.3,
scikit-learn 1.8.0 and threadpoolctl 3.6.0:

- The complete source suite collected **600 tests: 594 passed, 6 skipped**,
  with zero failures or errors. The six external semiconductor-toolbox
  walkthrough integration tests were skipped because a separate trusted
  matching toolbox checkout was not supplied; they were **not** passed.
- A separate offline installation of this wheel passed **72 report/CLI/init
  tests with zero skips**, plus all four CLI version checks and static
  regression/binary report checks. Imports came from the installed package.
- Two clean local builds reproduced the wheel and source archive byte-for-byte
  after source-archive metadata normalization. Archive payloads, wheel RECORD
  hashes and bundled license notices were verified.

These source-suite and installed-wheel checks overlap; do not add their counts.
They are author software checks, not independent acceptance or real-data
experiments. Full HTML browser rendering,
screen-reader testing and browser print-layout inspection were not performed;
static report structure and isolated vector proofs do not certify those flows.

Native Windows/macOS, Python 3.10, minimum-dependency combinations and hosted CI
remain unverified. A `py3-none-any` wheel does not certify native-platform
behavior. Verification of local artifacts does not establish downloaded release
binary byte verification or publication.

## Filesystem safety, licensing and privacy

Use init only in **trusted, idle local directories**. Failed creation can leave
partial output; retry a fresh directory name. A status file is not proof that a
failed command succeeded. This is **not a filesystem transaction or comprehensive
TOCTOU defense**: hostile concurrent ancestor replacement can create new files
outside the intended path before detection. See the [setup guide](WORKFLOW_INIT.md).

Original code and invented fixtures remain MIT-licensed. Bundled code, artwork
and typography keep their existing notices. No user measurements, model weights,
telemetry, automatic uploads or hosted ML service are added. Review filled inputs
and reports before sharing: they can contain IDs, measurements, targets and
predictions. Passing software tests does not establish real-device prediction
quality, physical validity, causal identification or calibrated uncertainty.

## 繁體中文：版本與使用範圍

b5 是報告與文件的視覺維護：統一線圈徽記、流程圖、深色／行動版及雙語導覽。
不是新的 ML 能力，也不是科學驗證。模型、綱要、資料切分、指標、相依套件、
量測行為及虛構範例不變。b4 引入的空白輸入 init 與 b3 引入的 FET 繼續保留；
舊版指南、tag 與發布資產保持凍結。

安裝前確認 b5 發布頁確有相符的 wheel、原始碼與完整校驗碼。候選檔案不等於
已發布。四個版本指令應回報 `0.3.0b5`；init 說明中的 b4 是功能首次提供版本。
作者完整原始碼測試共 600 項：594 通過、6 項外部 toolbox 整合測試因未提供
獨立且相符的可信 checkout 而略過，不算通過。另有已安裝 wheel 的 72 項
報告／CLI／init 測試通過、沒有略過；兩組涵蓋重疊，不應相加。
視覺驗證限靜態結構與先前的獨立向量圖證據，未涵蓋瀏覽器、螢幕閱讀器或列印排版；
原生 Windows／macOS、Python 3.10、最低相依版本組合及託管 CI 尚未驗證。
只在可信且沒有其他寫入者的本機目錄操作；既有 TOCTOU 限制不變。
