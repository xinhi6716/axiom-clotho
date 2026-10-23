"""Original synthetic binary software demo; not a semiconductor benchmark."""
from groupbench import (make_binary_fixture, plan_binary_benchmark, benchmark_binary,
                        fit_predict_binary)

frame = make_binary_fixture()  # 1 = fictitious fail, 0 = fictitious pass
options = dict(target="fail", group="lot")
print(plan_binary_benchmark(frame, **options)["fit_budget"])
result = benchmark_binary(frame, **options)
print(result.leaderboard[["model", "group_brier", "brier"]])
result.write("runs/binary-python-evaluation")  # Must be fresh; no overwrite.
new_rows = frame.drop(columns=["lot", "fail"]).iloc[:5].copy()
prediction = fit_predict_binary(frame, new_rows, **options)
print(prediction.predictions)
prediction.write("runs/binary-python-prediction")
# These reused invented feature rows illustrate the interface only. They are
# not independent observations or evidence about deployment performance.
