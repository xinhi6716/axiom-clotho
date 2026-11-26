# The last feedback to arrive may not describe the latest event

Run `python examples/delayed-feedback-recency/run.py` from the checkout.
Two invented events have outcomes 0 and 1 at event ticks 1 and 2.
Their labels arrive at ticks 5 and 3 respectively. A prediction can use
a label only if its arrival is strictly before the prediction cutoff.
Both trailing-one update rules comply with that constraint. At cutoff
4 both return 1; at cutoff 6 the last-arrival rule rewinds to 0 while
the most recent available event rule stays at 1. For the two invented
later outcomes (both 1), Brier scores are 1/2 and 0; a frozen forecast
of 1/2 scores 1/4. Neither rule is claimed universally better.

[Joulani, György and Szepesvári (2013), original ICML paper](https://proceedings.mlr.press/v28/joulani13.html)
studies learning under delayed feedback. This demonstration does not
implement their algorithms or claim a regret or coverage guarantee.
It updates predictions prospectively, unlike a fixed-roster exercise
about which outcomes are available to score.

If an update rule means recency of events, record event and availability
clocks separately, pick the eligible event by the former only after
filtering by the latter, and audit delayed old arrivals in an untouched
replay. Freeze candidate and static baseline and never rewrite forecasts
already issued. Use a declared sequencer for simultaneous events if the
real clock has ties; this toy uses distinct event and arrival ticks.
