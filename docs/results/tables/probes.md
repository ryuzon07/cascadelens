# Leakage probes

| Probe | Result | Outcome |
| --- | --- | --- |
| Identity | drop, graph and structural shuffled: 0.2 [-1.5, 1.9] | pass |
| Graph-only | Top-1 group_run 12.3 [9.0, 15.8], loso 7.4 [5.5, 9.4] | pass |
| Window leak | no cliff at zero; 98.9% of injections sit within 2% of the window's middle | pass |
| Mask-only | Top-1 31.5 [22.9, 40.7] on clean data, uniform floor 4.4 | flagged |
