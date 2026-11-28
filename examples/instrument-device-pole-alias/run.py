"""Exact cascade-identifiability witness using only invented values."""

from fractions import Fraction as F
import json


def response(pole, signal):
    """Zero-state, time-invariant discrete first-order response."""
    if type(pole) not in (F, int) or not 0 <= pole < 1:
        raise ValueError("need an exact rational pole in [0, 1)")
    state = F(0)
    out = []
    for x in signal:
        if type(x) not in (F, int):
            raise ValueError("need exact rational inputs")
        state = pole*state + (1-pole)*x
        out.append(state)
    return out


def cascade(device, sensor, signal):
    return response(sensor, response(device, signal))


def witness():
    a,b,new_path=F(3,4),F(1,4),F(1,2)
    fit = [1]*4
    observed = cascade(a,b,fit)
    if observed != cascade(b,a,fit):
        raise ValueError("a shared cascade must commute in this control")
    held = [1,1]
    truth, swapped = cascade(a,new_path,held), cascade(b,new_path,held)
    return {
        "same_training_cascade": list(map(str,observed)),
        "two_sensor_only_initial_responses": [str(response(p,[1])[0]) for p in (b,a)],
        "new_path_slow_device": list(map(str,truth)),
        "new_path_fast_device": list(map(str,swapped)),
        "wrong_device_attribution_mae": str(sum(abs(x-y) for x,y in zip(truth,swapped))/len(held)),
    }


if __name__=="__main__":
    print(json.dumps(witness(),sort_keys=True))
