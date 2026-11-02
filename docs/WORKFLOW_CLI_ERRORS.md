[Axiom Clotho](../README.md) / Workflow command guidance

# Correct a workflow command without exposing its values

In this source checkout, syntax errors for `clotho init`, `prepare`, `plan`,
`run`, `score`, `demo`, and the root command show the declared usage and the
matching `--help` command. They return exit status **2**, write guidance to
**stderr**, and leave stdout empty. They do not run a workflow stage or create
an output directory. This does not change a previously built wheel; use the
source containing this change or a later distribution that includes it.

For example, `clotho run` is missing its required arguments:

```text
usage: clotho run [-h] --prepared PREPARED --output-dir OUTPUT_DIR
error: invalid workflow arguments; use `clotho run --help` for supported commands and options. Supplied values are not echoed.
```

The usage line may wrap to fit the terminal. `PREPARED` and `OUTPUT_DIR` are
placeholders for your local directories, not literal filenames. Run the
suggested help command to inspect the options, then correct the original command.
No model fitting or data loading happens while showing this syntax guidance.

## Required inputs at a glance

| Command | Required options |
| --- | --- |
| `clotho init` | `--workflow rram` or `--workflow fet`, and `--out` |
| `clotho prepare` | `--input-dir`, `--partitions`, `--training-targets`, `--config`, `--output-dir` |
| `clotho plan` | `--prepared` |
| `clotho run` | `--prepared`, `--output-dir` |
| `clotho score` | `--run-dir`, `--outcomes`, `--output-dir` |
| `clotho demo` | `--output-dir`, `--run-invented-example` |

The `mlcalc-workflow` alias and `python -m mlworkflow` use the same parser;
help displays the canonical `clotho` command name. Existing option names,
required inputs, successful outputs, and exit statuses are unchanged. Calling
`clotho` with no arguments still displays help and does not start a demo.

## Privacy and scope

Syntax guidance is built from declared option names and placeholders, not the
original error string. Unknown options, invalid choices, argument values,
control characters in arguments, and the executable's local pathname are not
echoed in these parser errors. As before, this does **not** redact your shell
history or process arguments. Review commands and outputs before sharing them.

This guidance distinguishes syntax mistakes from component/resource failures.
A correctly formed command can still fail validation, encounter unavailable
local resources, or leave an incomplete output after an I/O interruption.
Follow that stage's error and bundle contract; a usage message is not scientific
validation and retrying is not a reason to overwrite an existing output.

The delegated `clotho tabular` and `clotho features` commands retain their own
parsers and error behavior. This change does not promise their errors are
redacted, modify upstream components, or expand supported input formats.

See [blank-input setup](WORKFLOW_INIT.md) for the measurement checklist and
[the RRAM workflow](CLOTHO_CSV_BETA.md) or
[the FET workflow](FET_DEVICE_REGRESSION.md) for the actual data contracts.
