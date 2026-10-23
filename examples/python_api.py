"""Run from an installed checkout; all data here is synthetic."""

from pathlib import Path

from groupbench import GroupBenchRegressor, benchmark, make_materials_fixture


def main():
    data = make_materials_fixture()
    result = benchmark(data, target="response", group="batch", random_state=7)
    print(result.leaderboard.to_string(index=False))
    print(result.write(Path("runs/python-api")))

    # After evaluation, refit a deployable estimator on all available data.
    # Its training fit is NOT another generalization estimate.
    X = data.drop(columns=["response", "batch"])
    estimator = GroupBenchRegressor(random_state=7)
    estimator.fit(X, data["response"], groups=data["batch"])
    new_process = X.iloc[:2].copy()  # Shape-only illustration, not new observations.
    print("Illustrative predictions:", estimator.predict(new_process))
    print("Inner-selected candidate:", estimator.best_candidate_)


if __name__ == "__main__":
    main()
