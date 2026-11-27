# A section center does not certify the sampled window

```sh
python examples/steady-regime-window-containment/run.py
```

[Ueda et al., §5.2](https://www.jstage.jst.go.jp/article/jamdsm/14/1/14_2020jamdsm0015/_article)
separately examined the response of powder supply to a changed command.
This motivates keeping the command and the material-delivery regime
distinct. Every value below is invented; no lag or device setting is
taken from the paper.

Suppose a synthetic path has coordinate s(t)=2t and a command interval
[2,10]. An external certification is assumed to establish the *entire
steady interval*, from time 2 plus an entry delay in [1,3] to time 10 plus
an exit delay in [0,2]. The example assumes exact clocks and speed, closed
certified steady boundaries, and all combinations of the two delay bounds
and a common section registration offset in [−1,1] are possible.
Neither a command timestamp nor first detected material would establish
that certificate.

The midpoint-delay spatial regime [8,22] contains every nominal section
center, but the spatial intersection of all certified regimes is [10,20].
For measured window [l,r], check its complete registration envelope
[l−1,r+1]. Windows [10,12] and [19,20] fail certification, while [11,12]
and [13,17] pass. A passing boundary counts only because closed boundaries
are certified here. The four delay endpoint pairs and two offset endpoints
suffice for a worst-case test of this affine toy: the constraints are
linear and the extrema are attained at corners. Failure counts are
numbers of compatible corner worlds, **not probabilities**.

Use a matched metrology window and validated response and registration
bounds before assigning one steady recipe to a full section. An unresolved
window is not proof of transient material or an instruction to interpolate.
If the path is nonlinear, reverses, or errors are dependent and the
rectangle is not exact, rederive the feasible envelope.
