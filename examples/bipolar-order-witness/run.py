"""Equal signed exposure summaries do not determine state in a clipped toy."""
from itertools import product
import json


def summary(commands):
    if any(command is None for command in commands):
        return None
    return {
        "net": sum(commands),
        "positive": commands.count(1),
        "negative": commands.count(-1),
    }


def replay(commands):
    if any(command is None for command in commands):
        raise ValueError("polarity unresolved")
    state = 0
    for command in commands:
        state = max(0, min(2, state + command))
    return state


def possible_states(commands):
    unknown = sum(command is None for command in commands)
    states = set()
    for signs in product((-1, 1), repeat=unknown):
        rest = iter(signs)
        states.add(replay(tuple(next(rest) if c is None else c for c in commands)))
    return sorted(states)


def fixture():
    early_plus = (1, 1, -1, -1)
    late_plus = (-1, -1, 1, 1)
    unknown = (1, -1, None)
    return {
        "early_plus": {"exposure": summary(early_plus), "state": replay(early_plus)},
        "late_plus": {"exposure": summary(late_plus), "state": replay(late_plus)},
        "unknown_polarity": {"exposure": summary(unknown),
                             "conditional_possible_states": possible_states(unknown)},
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), indent=2))
