# Start with your own measurements

**Included in 0.3.0b4. The frozen 0.3.0b3 wheel and tag do not include
`clotho init`.** Install the matching b4 wheel or source as described in the
[b4 release guide](RELEASE_NOTES_0.3.0b4.md). Existing RRAM/FET examples and commands
remain unchanged. From the matching checkout, `PYTHONPATH=src python -S -m mlworkflow`
can replace `clotho`. Do not install a second overlapping distribution or mistake
a local candidate wheel for proof that a release has been published.

## One small setup step, then the existing four stages

```sh
clotho init --workflow rram --out my-measurements
# OR choose the separate FET input contract:
clotho init --workflow fet --out my-transfer-data
```

Choose a fresh directory whose parent already exists. Init rejects pre-existing
outputs and symlink path components at preflight; exclusive file creation preserves
competing files. It does not create missing parents. There is no `--force`.
It generates no measurement rows, endpoints,
predictions or scientific results. No account, download, extraction, model fit,
telemetry or hosted service is involved; init and its help need only Python's
standard library.

Open the generated **README.md** first. It contains the complete local checklist,
units, grouping and acquisition rules, editable configuration choices, command
sequence and a Traditional Chinese quick guide. Its contract links are optional
reading, not resources fetched by the command.

| Generated files | Purpose |
| --- | --- |
| `raw/manifest.csv`, `raw/conditions.csv`, `raw/iv_sweeps.csv` (RRAM) | Exact existing RRAM headers, with no observations. |
| `raw/transfer.csv` (FET) | Exact existing FET header, with no observations. |
| `partitions.csv` | Declare every physical device as `train` or `predict`. |
| `training_targets.csv` | Supply independently measured targets for training devices only. |
| `later_outcomes.csv` | Leave empty until later scoring; never use as a training input. |
| `config.json` | Existing schema, with scientific choices deliberately unfinished. |
| `README.md` | Self-contained next steps, even without a source checkout. |
| `SCAFFOLD_STATUS.json` | Creation status only; never edit it to imply success. |

RRAM keys are globally namespaced device IDs. FET keys are exact
`(lot, wafer, device)` tuples, not device labels alone. Keep IDs as text and retain
leading zeros and case. Do not silently drop, coerce or join rows by position.

## Choices that are deliberately not made for you

Endpoint name, target unit and physical grouping start as JSON `null`; features
start as an empty list. RRAM's bracket span and every FET measurement parameter
also start as `null`. These are **invalid until you supply justified choices**.
All four scientific declarations start as `false`. Only change them if your
actual protocol supports the assertions, never merely to make validation pass.

- **RRAM:** the existing format fixes voltages to V, currents to A, temperature
  to K and read delay to ms. The scaffold documents those requirements; it does
  not infer the units of your source or convert instrument exports. Select batch,
  wafer or device grouping and a justified positive bracket span in volts.
- **FET:** explicitly choose voltage/current and criterion units, signed criterion
  current, polarity, positive bracket span and ordered window bounds. Exactly one
  complete acquisition branch is accepted per composite physical device. No
  averaging, branch selection or physical-regime discovery occurs.
- Computational starting settings reuse the public examples: RRAM uses 3 outer /
  2 inner splits, seed 7 and max_fits 200; FET uses 2 / 2, seed 1729 and max_fits
  500. Both use dummy/ridge and group_mae. These are editable, not scientific
  recommendations; the actual plan must validate available groups and fit counts.

The **untouched scaffold fails `prepare` before any prepared output is written**.
Fill it with your own measurements and justified configuration, then follow its
README: prepare → inspect plan → decide whether to run → score later outcomes.
The existing [RRAM](CLOTHO_CSV_BETA.md) and [FET](FET_DEVICE_REGRESSION.md)
contracts remain the only preparation routes. This adds setup, not another model.

## Blank setup is not an invented demonstration

For a complete software example, use the existing explicitly invented
[RRAM example](../examples/measurement-regression/README.md) or
[FET example](../examples/fet-device-regression/README.md). Their numbers and
declarations are demonstration inputs, not recommended settings or real evidence.
`clotho demo` and `clotho tabular fixture` also remain separate invented-data tools.
Init never copies their measurement rows or automatically accepts their declarations.

## Failure and privacy boundaries

Init uses private-mode files/directories where POSIX permissions apply. An I/O
interruption may leave partial output. `SCAFFOLD_STATUS.json` starts with
`INCOMPLETE_DO_NOT_USE`; missing or malformed status is also incomplete. Success
records `READY_TO_EDIT_NOT_VALIDATED`. Status updates use only the still-open
handle of the exclusively created status file, never path-based unlink or replace.
Nothing is deleted automatically. Observed root/status identity replacement,
unexpected entries or changed generated bytes fail creation. Retry a fresh name;
do not edit status to imply success. A failed command is a failure even if a status
file appears ready. No `COMPLETE.json` is created: this is not an analysis bundle.

Use trusted, idle local directories: this is not a filesystem transaction or a
guarantee against hostile concurrent directory replacement, which can still cause
outside file creation before a replacement is detected. Keep filled inputs
and outputs private; review all IDs, measurements, targets and predictions before
any separately authorized sharing. A scaffold, configuration or checksum cannot
establish scientific validity, correct units, physical independence or blinding.

## 繁體中文快速指南

**0.3.0b4 已包含 `clotho init`；凍結的 0.3.0b3 wheel 與 tag 沒有此功能。**
請安裝相符的 b4 wheel 或原始碼；不必另裝重疊套件，也不必啟用雲端運算。

1. 用 `--workflow rram` 或 `--workflow fet` 明確選擇資料格式，並以 `--out`
   指定尚不存在的目錄；上層目錄須已存在，不接受符號連結路徑。
2. 先開啟產生的 `README.md`。CSV 只有標頭，沒有虛構資料；請填自己的量測。
3. 自行填寫目標名稱、目標單位、實體分組、特徵與量測參數。`null` 和空白特徵
   清單是刻意保留的阻擋項目，程式不會猜測。RRAM 固定要求 V、A、K、ms；
   FET 的電壓／電流單位、準則與視窗都必須明確指定。
4. 科學聲明預設 `false`。只有真實流程支持時才能改成 `true`，不可只為通過檢查。
5. 訓練目標只含 train 裝置。`later_outcomes.csv` 留待預測完成且後續結果可用，
   僅供 score 使用；FET 以完整 `(lot, wafer, device)` 身分精確連接。
6. 先 prepare，再 plan 檢查；確認後才 run，最後 score。每階段輸出使用全新相鄰
   目錄。未填寫的範本會在寫入 prepared 目錄之前被拒絕。

這是空白設定工具，不是完整示範，也不代表科學驗證。若初始化中斷，請保留失敗
狀態並換新目錄重試，不要刪除標記來假裝成功。真實輸入與結果請保持私密。
