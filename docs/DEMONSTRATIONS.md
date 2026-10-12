# Optional verified demonstrations

I use automatic movement discovery in default LEARN. I made expert recordings
optional and do not bundle them as guaranteed room solutions.

**TOOLS → Import demonstration** opens a JSON picker and runs the recording from a fresh start
of its specified chapter in an isolated Celeste worker. It changes no learned
weights. Only if Celeste reports actual chapter completion does it add successful
room routes to `training/success-memory`. The next LEARN run can imitate them.
An unfinished, invalid, interrupted, or uncompleted chapter is not accepted.

The format is:

```json
{
  "chapter": 0,
  "actions": [
    [2, 1, 0, 0, 0, 0, 2],
    [2, 1, 1, 0, 0, 0, 3]
  ]
}
```

This is a format illustration, **not a complete Prologue recording**.

| Position | Choices |
| --- | --- |
| 0: horizontal | 0 left, 1 neutral, 2 right |
| 1: vertical | 0 up, 1 neutral, 2 down |
| 2–5: jump, dash, grab, interact | 0 released, 1 held |
| 6: duration index | 0–6 mean 1, 2, 4, 8, 12, 20, 30 simulation frames |

Use integer values. Chapter IDs are 0 through 7. Input lists are limited to
100,000 actions; stored room routes to 2,048 decisions. The verifier uses actual
consumed action frames, which can end early on room changes/deaths. A recording
must follow the same bridge semantics. It does not parse CelesteTAS files or
capture manual play automatically; those would require a separate conversion
or recorder implementation.

Stop training before importing; the tool shares the package's run lock. STOP
also interrupts verification. Successful imports are labeled
`verified_demonstration`, distinct from `agent_room_clear` and `search_room_clear`.
Published results should disclose demonstrations and training search assistance.
