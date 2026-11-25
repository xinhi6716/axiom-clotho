# Pairwise balance can hide a missing process contrast

```sh
python examples/cyclic-factorial-alias/run.py
```

An invented three-factor design has nine rows. Use levels 0, 1 and 2
for power, speed and nominal powder feed, but at each fixed power and
speed use only the powder level determined by their sum modulo 3.
Each two-factor margin contains every pair once. Nevertheless no fixed
power × speed cell has two powder levels.

Two response rules agree on every observed row: one responds only to
powder, the other only to power × speed. At fixed power and speed
(0,0), changing powder from 0 to 2 gives differences of 2 and 0,
respectively. Adding that prespecified contrast creates support where
there was none; independent fabrication blocks are still needed to
separate a repeatable effect from run variation.

This is an exact synthetic design witness, not an audit of model fits
or a claim that orthogonal arrays cannot estimate additive main effects.
The two rules differ because an arbitrary cell interaction is allowed.
Do not count balanced two-factor margins as a matched physical contrast.
