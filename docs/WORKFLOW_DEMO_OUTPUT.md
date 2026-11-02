[Axiom Clotho](../README.md) / Demo output destinations

# Choose a fresh destination for the invented demo

`clotho demo` requires both an explicit invented-example acknowledgment and a
**new directory whose parent already exists**:

```sh
clotho demo --output-dir invented-demo --run-invented-example
```

This command runs the existing invented semiconductor example and writes its
output. It is not a no-fit preview or evidence of real-device performance.
`clotho demo --help` only displays its options.

## Correct the destination, not the installation

In source versions containing this change, these known preflight failures have
separate recovery guidance:

| Failure | What to do |
| --- | --- |
| The destination already exists as a file, directory, or symlink, including a dangling symlink. | Choose a fresh `--output-dir`. Existing contents are not overwritten. |
| The destination's parent does not exist or is not a directory. | Check the parent or choose another destination with an existing parent. Missing parents are not created automatically. |

For example, if `invented-demo` already exists, the error is:

```text
error: demo output already exists; choose a fresh --output-dir. Existing files and directories are not overwritten.
```

An empty existing directory is still an existing destination. There is no demo
`--overwrite` option. Do not delete or empty an earlier result merely to silence
the error; use a different destination and retain incomplete results for diagnosis.
This source change does not alter an already built wheel.

## What a preflight rejection guarantees

These two checks run before component verification, calculator-source staging,
or subprocess execution. A rejection returns exit status **2**, prints a static
message to **stderr**, leaves stdout empty, and makes no workflow output writes.
The message does not contain the supplied path or control characters from it.
Correcting a destination does not by itself establish that dependencies,
components, data, or the later workflow will succeed.

Other component and operating-system errors retain the existing generic,
redacted resource handling. Syntax mistakes, such as omitting
`--run-invented-example`, retain the separate
[command-specific usage guidance](WORKFLOW_CLI_ERRORS.md).

## Boundaries that have not changed

- Preflight is an early check, not a directory reservation or a filesystem
  transaction. If another process creates the destination afterward, the
  existing runner still rejects an already present destination. Use trusted,
  idle local directories; hostile ancestor replacement is not prevented here.
- A parent that is a symlink to an existing directory remains accepted by the
  demo's existing preflight. Do not confuse this with `clotho init`, which has
  a different, stricter path contract.
- Preflight does not test parent write permission or promise successful later
  writes. A later interruption can leave an incomplete bundle. Do not treat it
  as complete or reuse its directory for another attempt.
- Successful invocation still delegates to the same pinned example runner,
  with unchanged options, output content, and subprocess exit status.
- Redacted diagnostics do not redact shell history or process arguments.
  Review commands, logs, and output files before sharing them.

For your own measurements rather than an invented demo, start with
[blank-input setup](WORKFLOW_INIT.md) and its explicit scientific choices.
