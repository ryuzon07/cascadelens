The first run read `data.csv`, as the plan said, and failed on two systems:

| System | Published | Ours on `data.csv` | Difference |
| --- | --- | --- | --- |
| RE1-OB | 0.906 | 0.896 | -0.010 |
| RE1-SS | 0.954 | 0.576 | -0.378 |
| RE1-TT | 0.826 | 0.261 | -0.565 |

**Cause: the input file.** Online Boutique matched and is the one system with no
`simple_data.csv`. For Sock Shop and Train Ticket, `data.csv` holds 438 and 1,242 columns, including
node exporters and cumulative counters, against 60 and 221 in `simple_data.csv`. `main.py` has a
commented-out switch to `simple_data.csv`, so the published table evidently used it, although
neither paper says so. This run reads that file where it exists. BARO's own logic is unchanged.

**Remaining cell gap.** Train Ticket DELAY is 0.58 against 0.77 published, past the tolerance for a
single cell. The system mean still passes, at -0.047, close to the limit. The cause was not
established; library drift and the unstated data length are the candidates in the plan.

**Consequence for the evaluation run.** Our cases are built from `data.csv`, so `baro` there sees
the harder metric set, and its scores should not be compared with the published numbers.
