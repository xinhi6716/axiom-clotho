# Audit timing separately from operation counts

```sh
python examples/operation-time-ledger/run.py
```

These three invented metadata rows are not device measurements. Every
timestamp is on one declared clock. A and B independently illustrate
overlap and reference age; both declare success at tick 60 and are
adjudicated at tick 1060. In this ledger, declaration ends programming writes;
service reads can continue until the separately logged terminal timestamp. The
reference read is independent and is not in the deployment operation list.

| Attempt | Active operation widths added | Union of operation intervals | Start to terminal | Since last write at reference | Since declaration at reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 40 | 35 | 60 | 1020 | 1000 |
| B | 20 | 20 | 60 | 1050 | 1000 |
| C | 5 | 5 | not observed | not defined | not observed |

A's two operations are on separate resources in this illustration. Summed
width is work; the union is the time with at least one active operation
on the declared clock. The gap between union and elapsed time is not an
attribution to any particular source of overhead. Matching time since
declaration does not match time since last physical write. C has no
recorded write, adjudication or terminal time. Keep it in the initiated
roster; do not silently turn an unknown duration into zero or report a
complete-cohort mean from A and B. A real protocol should define expected
clock alignment, interruption/censoring rules and a common reference
anchor before evaluation. This example fits nothing, estimates no
energy, and does not rank real policies.
